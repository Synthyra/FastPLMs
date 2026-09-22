"""Modal workers that run one fixed FastPLMs test stage and report its outcome."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import modal

from pathlib import Path

from .config import (
    APP_NAME,
    CPU_CORES,
    CPU_MEMORY_GIB,
    CPU_WHEEL_INDEX,
    CPU_WORKER_TIMEOUT_SECONDS,
    CUDA_WHEEL_INDEX,
    ENVIRONMENT_PROBE_TIMEOUT_SECONDS,
    FOLD_WORKER_TIMEOUT_SECONDS,
    GPU_CHOICES,
    GPU_CORES,
    GPU_MEMORY_GIB,
    GPU_WORKER_TIMEOUT_SECONDS,
    OUTPUT_TAIL_CHARACTERS,
    PYTHON_VERSION,
    STARTUP_TIMEOUT_SECONDS,
    TORCH_VERSION,
    TRANSFORMERS_VERSION,
)
from .source import (
    BASELINE_DIRECTORY,
    BASELINE_WORKSPACE,
    ROOT,
    SOURCE_DIRECTORIES,
    SOURCE_FILES,
    WORKSPACE,
    excluded_from_upload,
)
from .stages import STAGES, stage_arguments


CACHE_ROOT = "/vol"
# Set on the stage subprocess, not in the image. Modal refuses to mount a volume
# on a non-empty path, and an image-level cache variable under the mount point
# let a build step write there.
CACHE_ENVIRONMENT = {
    "HF_HOME": f"{CACHE_ROOT}/huggingface",
    "TORCH_HOME": f"{CACHE_ROOT}/torch",
    "XDG_CACHE_HOME": f"{CACHE_ROOT}/xdg",
}
# Reported by a short child process so the worker itself never holds a CUDA
# context while the stage under test is running.
_ENVIRONMENT_PROBE = """
import json, platform, torch, transformers
cuda = torch.cuda.is_available()
print(json.dumps({
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if cuda else None,
    "compute_capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
}))
"""

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name(APP_NAME, create_if_missing=True)
# Forward a credential only when the launcher's environment defines it. Names
# are tested for presence; values are never read here.
credentials = modal.Secret.from_local_environ(
    [name for name in ("HF_TOKEN",) if name in os.environ]
)


# Mirrors the ``reference`` stage of docker/esmfold2-validation.Dockerfile: the
# official ESMFold2 stack needs its own Transformers fork and Hub client, so it
# lives in a separate interpreter beside the candidate environment.
REFERENCE_ENVIRONMENT = "/opt/reference"
_REFERENCE_SOURCES = "/opt/reference-src"
_REFERENCE_EXTRA_PACKAGES = (
    "attrs pandas cloudpathlib httpx tenacity zstd scikit-learn boto3 pygtrie "
    "dna_features_viewer pydssp ipython"
)
_FORK_PATHS_NOT_INSTALLED = [
    ".git",
    ".github",
    "tests",
    "docs",
    "examples",
    "benchmark",
    "benchmark_v2",
    "notebooks",
    "templates",
    "i18n",
    "docker",
    "scripts",
    "cookbook",
    "_assets",
]


def _with_reference_environment(image: modal.Image) -> modal.Image:
    """Install the pinned official ESMFold2 stack into its own interpreter."""
    pip = f"{REFERENCE_ENVIRONMENT}/bin/pip install --no-cache-dir"
    requirements = f"{_REFERENCE_SOURCES}/requirements"
    return (
        image.apt_install("build-essential")
        .add_local_dir(str(ROOT / "requirements"), requirements, copy=True)
        .add_local_dir(
            str(ROOT / "vendor/upstream/biohub-transformers"),
            f"{_REFERENCE_SOURCES}/transformers",
            copy=True,
            ignore=_FORK_PATHS_NOT_INSTALLED,
        )
        .add_local_dir(
            str(ROOT / "vendor/upstream/biohub-esm"),
            f"{_REFERENCE_SOURCES}/esm",
            copy=True,
            ignore=_FORK_PATHS_NOT_INSTALLED,
        )
        .run_commands(
            f"python -m venv {REFERENCE_ENVIRONMENT}",
            f"{pip} torch=={TORCH_VERSION} --index-url {CUDA_WHEEL_INDEX}",
            f"{pip} -r {requirements}/core.in -r {requirements}/features/structure.in "
            f"-c {requirements}/constraints/validation.txt {_REFERENCE_EXTRA_PACKAGES}",
            f"{pip} --no-deps {_REFERENCE_SOURCES}/transformers {_REFERENCE_SOURCES}/esm",
            f"{pip} huggingface-hub==0.36.0",
        )
    )


def _image(
    torch_index: str, feature_files: tuple[str, ...], *, reference_environment: bool = False
) -> modal.Image:
    """Build one validation image from the pinned requirement files."""
    image = (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .apt_install("git", "libgomp1")
        .uv_pip_install(f"torch=={TORCH_VERSION}", index_url=torch_index)
        .pip_install_from_requirements(str(ROOT / "requirements/core.in"))
    )
    # Profiles use relative ``-r`` includes, so install each feature file directly.
    for feature_file in feature_files:
        image = image.pip_install_from_requirements(
            str(ROOT / "requirements/features" / feature_file)
        )
    image = image.uv_pip_install(f"transformers=={TRANSFORMERS_VERSION}")
    # Build steps must precede the working-tree mounts below.
    if reference_environment:
        image = _with_reference_environment(image)
    image = image.env(
        {
            # Mirrors the ``x-common`` environment in docker/compose.yaml.
            "PYTHONPATH": f"{WORKSPACE}/src:{WORKSPACE}",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "FASTPLMS_ARTIFACTS": f"{WORKSPACE}/artifacts",
        }
    ).workdir(WORKSPACE)
    for directory in SOURCE_DIRECTORIES:
        image = image.add_local_dir(
            str(ROOT / directory), f"{WORKSPACE}/{directory}", ignore=excluded_from_upload
        )
    for file_name in SOURCE_FILES:
        image = image.add_local_file(str(ROOT / file_name), f"{WORKSPACE}/{file_name}")
    # The launcher exports this before import; inside a worker the local path is absent.
    if BASELINE_DIRECTORY.is_dir():
        image = image.add_local_dir(
            str(BASELINE_DIRECTORY), BASELINE_WORKSPACE, ignore=excluded_from_upload
        )
    return image


# requirements/profiles/cpu-validation.in and candidate.in, plus structure extras.
cpu_image = _image(CPU_WHEEL_INDEX, ("dev.in", "structure.in", "train.in"))
gpu_image = _image(CUDA_WHEEL_INDEX, ("dev.in", "flash.in", "structure.in", "train.in"))
fold_image = _image(
    CUDA_WHEEL_INDEX,
    ("dev.in", "flash.in", "structure.in", "train.in"),
    reference_environment=True,
)


def _run_stage(stage_name: str, selection: str | None) -> dict[str, object]:
    """Run one stage from the fixed table and describe what happened."""
    started = time.monotonic()
    spec = STAGES[stage_name]
    junit_path = Path("/tmp/junit.xml")
    arguments = stage_arguments(spec, selection, str(junit_path))
    environment = json.loads(
        subprocess.run(
            [sys.executable, "-c", _ENVIRONMENT_PROBE],
            capture_output=True,
            text=True,
            check=True,
            timeout=ENVIRONMENT_PROBE_TIMEOUT_SECONDS,
        ).stdout
    )
    timed_out = False
    try:
        completed = subprocess.run(
            [sys.executable, *arguments],
            capture_output=True,
            text=True,
            timeout=spec.timeout_seconds,
            cwd=WORKSPACE,
            env={**os.environ, **CACHE_ENVIRONMENT},
        )
        exit_code: int | None = completed.returncode
        output = completed.stdout + completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out, exit_code = True, None
        output = _decoded(error.stdout) + _decoded(error.stderr)
    cache_volume.commit()
    return {
        "stage": stage_name,
        "selection": selection,
        "status": "passed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "timed_out": timed_out,
        "arguments": list(arguments),
        "environment": environment,
        "output_tail": output[-OUTPUT_TAIL_CHARACTERS:],
        "junit_xml": junit_path.read_text(encoding="utf-8") if junit_path.exists() else None,
        "elapsed_seconds": time.monotonic() - started,
    }


def _decoded(stream: bytes | str | None) -> str:
    if stream is None:
        return ""
    return stream if isinstance(stream, str) else stream.decode("utf-8", errors="replace")


cpu_worker = app.function(
    name="cpu_worker",
    image=cpu_image,
    cpu=CPU_CORES,
    memory=int(CPU_MEMORY_GIB * 1024),
    timeout=CPU_WORKER_TIMEOUT_SECONDS,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    volumes={CACHE_ROOT: cache_volume},
    max_containers=1,
)(_run_stage)

gpu_workers = {
    gpu: app.function(
        name=f"gpu_worker_{gpu}",
        image=gpu_image,
        gpu=gpu,
        cpu=GPU_CORES,
        memory=int(GPU_MEMORY_GIB * 1024),
        timeout=GPU_WORKER_TIMEOUT_SECONDS,
        startup_timeout=STARTUP_TIMEOUT_SECONDS,
        volumes={CACHE_ROOT: cache_volume},
        secrets=[credentials],
        max_containers=1,
        scaledown_window=2,
    )(_run_stage)
    for gpu in GPU_CHOICES
}

fold_workers = {
    gpu: app.function(
        name=f"fold_worker_{gpu}",
        image=fold_image,
        gpu=gpu,
        cpu=GPU_CORES,
        memory=int(GPU_MEMORY_GIB * 1024),
        timeout=FOLD_WORKER_TIMEOUT_SECONDS,
        startup_timeout=STARTUP_TIMEOUT_SECONDS,
        volumes={CACHE_ROOT: cache_volume},
        secrets=[credentials],
        max_containers=1,
        scaledown_window=2,
    )(_run_stage)
    for gpu in GPU_CHOICES
}
