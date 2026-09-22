"""Modal workers for isolated confidence tests, data preparation, and training."""

from __future__ import annotations

import json
import subprocess
import sys
import time

import modal

from pathlib import Path

from .config import (
    CAMPAIGN_TIMEOUT_SECONDS,
    GPU_STARTUP_TIMEOUT_SECONDS,
    TRAIN_TIMEOUT_SECONDS,
    VOLUME_NAME,
)


ROOT = Path(__file__).resolve().parents[2]
REMOTE_ROOT = Path("/vol/confidence")
app = modal.App("fastplms-confidence-pilot")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
credentials = modal.Secret.from_local_environ(["HF_TOKEN", "WANDB_API_KEY"])
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgomp1", "mmseqs2")
    .uv_pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install_from_requirements(str(ROOT / "requirements/core.in"))
    .pip_install_from_requirements(str(ROOT / "requirements/features/structure.in"))
    .uv_pip_install(
        "transformers==5.13.0",
        "pytest>=9,<10",
        "wandb==0.18.7",
        "lmdb==1.7.3",
        "gdown==5.2.0",
        "python-dotenv==1.1.1",
        "requests",
        "DockQ==2.1.3",
        "tmtools==0.2.0",
        "ruff==0.12.12",
        "numpy==1.26.4",
        "scipy==1.16.3",
        "gemmi==0.7.3",
        "pyarrow==21.0.0",
    )
    .env(
        {
            "PYTHONPATH": "/workspace/src:/workspace",
            "HF_HOME": "/vol/huggingface",
            "PYTHONUNBUFFERED": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "WANDB_DIR": "/vol/confidence/wandb",
            "WANDB_CONSOLE": "off",
        }
    )
    .workdir("/workspace")
    .add_local_dir(str(ROOT / "src"), "/workspace/src")
    .add_local_dir(str(ROOT / "tools"), "/workspace/tools")
    .add_local_dir(str(ROOT / "model_cards"), "/workspace/model_cards")
    .add_local_dir(str(ROOT / "docs"), "/workspace/docs")
    .add_local_dir(str(ROOT / "benchmarks"), "/workspace/benchmarks")
    .add_local_dir(str(ROOT / "LICENSES"), "/workspace/LICENSES")
    .add_local_dir(str(ROOT / "requirements"), "/workspace/requirements")
    .add_local_dir(str(ROOT / "docker/constraints"), "/workspace/docker/constraints")
    .add_local_dir(str(ROOT / "tests/parity/fixtures"), "/workspace/tests/parity/fixtures")
    .add_local_file(str(ROOT / "LICENSE"), "/workspace/LICENSE")
    .add_local_file(str(ROOT / "README.md"), "/workspace/README.md")
    .add_local_file(str(ROOT / "AGENTS.md"), "/workspace/AGENTS.md")
    .add_local_file(str(ROOT / "CLAUDE.md"), "/workspace/CLAUDE.md")
    .add_local_file(str(ROOT / "THIRD_PARTY_NOTICES.md"), "/workspace/THIRD_PARTY_NOTICES.md")
    .add_local_dir(str(ROOT / "tests/unit"), "/workspace/tests/unit")
    .add_local_dir(
        str(ROOT / "tests/fixtures/esmfold2_small"), "/workspace/tests/fixtures/esmfold2_small"
    )
    .add_local_file(str(ROOT / "pytest.ini"), "/workspace/pytest.ini")
    .add_local_file(str(ROOT / "tests/conftest.py"), "/workspace/tests/conftest.py")
    .add_local_file(str(ROOT / "ruff.toml"), "/workspace/ruff.toml")
)


@app.function(
    image=image,
    cpu=4,
    memory=32768,
    timeout=1800,
    startup_timeout=900,
    volumes={"/vol": volume},
    max_containers=1,
)
def cpu_stage(stage: str, options: dict) -> dict:
    start = time.monotonic()
    options = dict(options)
    options.pop("controller_app_id", None)
    options.pop("controller_call_id", None)
    REMOTE_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        if stage == "tests":
            paths = sorted(Path("tests/unit").glob("test_confidence_*.py"))
            paths += [
                Path("tests/unit/test_esmfold2_small.py"),
                Path("tests/unit/test_esmfold2_decode.py"),
            ]
            command = [sys.executable, "-m", "pytest", "-q", *map(str, paths)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=1200)
            report = {
                "status": "passed" if result.returncode == 0 else "failed",
                "exit_code": result.returncode,
                "output": result.stdout + result.stderr,
            }
        elif stage == "lint":
            paths = sorted(Path("tools/confidence").glob("*.py"))
            paths += sorted(Path("tests/unit").glob("test_confidence_*.py"))
            original = {str(path): path.read_text() for path in paths}
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ruff",
                    "check",
                    *(["--fix"] if options.get("fix") else []),
                    *map(str, paths),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            report = {
                "status": "passed" if result.returncode == 0 else "failed",
                "output": result.stdout + result.stderr,
                "formatted_files": {
                    str(path): path.read_text()
                    for path in paths
                    if original[str(path)] != path.read_text()
                },
            }
        elif stage == "format":
            paths = sorted(Path("tools/confidence").glob("*.py"))
            paths += sorted(Path("tests/unit").glob("test_confidence_*.py"))
            formatted = {}
            for path in paths:
                result = subprocess.run(
                    [sys.executable, "-m", "ruff", "format", "--stdin-filename", str(path), "-"],
                    input=path.read_text(),
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )
                if result.stdout != path.read_text():
                    formatted[str(path)] = result.stdout
            report = {"status": "formatted", "formatted_files": formatted}
        elif stage == "prepare":
            from .workflow import prepare_data

            report = prepare_data(REMOTE_ROOT, **options)
        elif stage == "docs":
            generator_options = []
            if options.get("esmc_report_root") is not None:
                generator_options.extend(["--esmc-report-root", str(options["esmc_report_root"])])
            if options.get("require_esmc_release_evidence"):
                generator_options.append("--require-esmc-release-evidence")
            generate = subprocess.run(
                [sys.executable, "-m", "tools.artifacts.generate_docs", *generator_options],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if generate.returncode != 0:
                raise RuntimeError(
                    "Documentation generation failed:\n" + generate.stdout + generate.stderr
                )
            check = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.artifacts.generate_docs",
                    *generator_options,
                    "--check",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if check.returncode != 0:
                raise RuntimeError(
                    "Generated documentation failed its check:\n" + check.stdout + check.stderr
                )
            generated_files = {}
            for directory in (Path("model_cards"), Path("docs/generated")):
                generated_files.update(
                    {
                        str(path): path.read_text(encoding="utf-8")
                        for path in sorted(directory.glob("*.md"))
                    }
                )
            report = {
                "status": "passed",
                "generated_files": generated_files,
                "generator_output": generate.stdout + generate.stderr,
                "check_output": check.stdout + check.stderr,
            }
        else:
            raise ValueError(f"Unknown CPU stage: {stage}")
        report["elapsed_seconds"] = time.monotonic() - start
        (REMOTE_ROOT / f"{stage}-report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    except Exception as error:
        # Preserve failed-stage evidence and measured runtime for the budget controller.
        report = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": time.monotonic() - start,
        }
        (REMOTE_ROOT / f"{stage}-report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        volume.commit()


def _gpu_stage(stage: str, model_id: str, options: dict) -> dict:
    started = time.monotonic()
    controller_app_id = options.pop("controller_app_id", None)
    controller_call_id = options.pop("controller_call_id", None)
    REMOTE_ROOT.mkdir(parents=True, exist_ok=True)
    (REMOTE_ROOT / "wandb").mkdir(exist_ok=True)
    try:
        import torch

        torch.use_deterministic_algorithms(True)
        from .workflow import run_gpu_stage

        if stage == "campaign":
            options["volume_commit"] = volume.commit
        report = run_gpu_stage(REMOTE_ROOT, stage, model_id, **options)
        report.update(
            {
                "controller_app_id": controller_app_id,
                "controller_call_id": controller_call_id,
            }
        )
        return report
    except Exception as error:
        report = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": time.monotonic() - started,
            "controller_app_id": controller_app_id,
            "controller_call_id": controller_call_id,
        }
        directory = REMOTE_ROOT / model_id
        directory.mkdir(exist_ok=True)
        (directory / f"{stage}-report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        volume.commit()


gpu_workers = {
    (gpu, timeout): app.function(
        name=f"gpu_{gpu}_{timeout}",
        image=image,
        gpu=gpu,
        cpu=4,
        memory=32768,
        timeout=timeout,
        startup_timeout=GPU_STARTUP_TIMEOUT_SECONDS,
        volumes={"/vol": volume},
        secrets=[credentials],
        max_containers=2,
        scaledown_window=2,
    )(_gpu_stage)
    for gpu, timeouts in {
        "L4": (600, 7200, TRAIN_TIMEOUT_SECONDS, CAMPAIGN_TIMEOUT_SECONDS),
        "L40S": (600, 7200, TRAIN_TIMEOUT_SECONDS, CAMPAIGN_TIMEOUT_SECONDS),
        "H100": (600, 7_200, TRAIN_TIMEOUT_SECONDS, CAMPAIGN_TIMEOUT_SECONDS),
    }.items()
    for timeout in timeouts
}


@app.function(image=image, cpu=0.25, memory=512, timeout=120, volumes={"/vol": volume})
def read_report(relative_path: str) -> dict:
    path = (REMOTE_ROOT / relative_path).resolve()
    if not path.is_relative_to(REMOTE_ROOT):
        raise ValueError("Report path escapes the pilot directory")
    return json.loads(path.read_text())
