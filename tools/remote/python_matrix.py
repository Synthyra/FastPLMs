"""Run the non-canonical FastPLMs source support matrix with uv."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


CANONICAL_GPU_PYTHON = "3.12"
PYTHON_SUPPORT_VERSIONS = ("3.11", "3.13", "3.14")
OFFLINE_SMOKE_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": "",
    "UV_TORCH_BACKEND": "cpu",
}


class MatrixCommandError(RuntimeError):
    """One uv or smoke command failed for a matrix member."""

    def __init__(self, stage: str, completed: subprocess.CompletedProcess[str]) -> None:
        super().__init__(f"{stage} failed with exit code {completed.returncode}")
        self.stage = stage
        self.completed = completed


def build_dependency_install_command(
    uv: str,
    python: Path,
    project_root: Path,
) -> tuple[str, ...]:
    """Build the CPU dependency installation command for one source smoke."""

    return (
        uv,
        "pip",
        "install",
        "--python",
        str(python),
        "--torch-backend=cpu",
        "-r",
        str(project_root / "requirements/profiles/runtime.in"),
        "-c",
        str(project_root / "requirements/constraints/validation.txt"),
    )


def build_smoke_environment(base: Mapping[str, str]) -> dict[str, str]:
    """Return the no-network, no-GPU environment used by each smoke."""

    environment = dict(base)
    environment.update(OFFLINE_SMOKE_ENVIRONMENT)
    return environment


def _run(
    stage: str,
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode:
        raise MatrixCommandError(stage, completed)
    return completed


def _output_fingerprint(text: str) -> dict[str, object]:
    """Describe subprocess output without persisting URLs, tokens, or host paths."""

    encoded = text.encode("utf-8", errors="replace")
    return {
        "bytes": len(encoded),
        "lines": len(text.splitlines()),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _run_member(
    *,
    uv: str,
    project_root: Path,
    temporary_root: Path,
    target: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    environment_root = temporary_root / f"python-{target.replace('.', '')}"
    environment = dict(os.environ)
    environment["UV_PYTHON_PREFERENCE"] = "only-managed"
    stage = "python-install"

    try:
        print(f"[{target}] installing the uv-managed interpreter", flush=True)
        _run(
            stage,
            (uv, "python", "install", target),
            cwd=project_root,
            environment=environment,
        )

        stage = "environment-create"
        print(f"[{target}] creating an isolated environment", flush=True)
        _run(
            stage,
            (uv, "venv", "--python", target, str(environment_root)),
            cwd=project_root,
            environment=environment,
        )
        python = environment_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.is_file():
            raise RuntimeError(f"uv did not create the expected interpreter: {python}")

        stage = "dependency-install"
        print(f"[{target}] installing the declared CPU runtime dependencies", flush=True)
        _run(
            stage,
            build_dependency_install_command(uv, python, project_root),
            cwd=temporary_root,
            environment=environment,
        )

        stage = "offline-cpu-source-smoke"
        print(f"[{target}] running the isolated repository-source smoke", flush=True)
        completed = _run(
            stage,
            (
                str(python),
                "-I",
                str(project_root / "tools/remote/python_support_smoke.py"),
                "--expected-python",
                target,
                "--source-root",
                str(project_root / "src"),
            ),
            cwd=temporary_root,
            environment=build_smoke_environment(environment),
        )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("The support smoke produced no JSON evidence.")
        evidence = json.loads(lines[-1])
        if not isinstance(evidence, dict):
            raise RuntimeError("The support smoke result is not a JSON object.")

        elapsed = time.perf_counter() - started
        return {
            "target": target,
            "status": "passed",
            "elapsed_seconds": round(elapsed, 3),
            "evidence": evidence,
            "stderr": _output_fingerprint(completed.stderr),
        }
    except MatrixCommandError as error:
        elapsed = time.perf_counter() - started
        return {
            "target": target,
            "status": "failed",
            "stage": error.stage,
            "elapsed_seconds": round(elapsed, 3),
            "returncode": error.completed.returncode,
            "stdout": _output_fingerprint(error.completed.stdout),
            "stderr": _output_fingerprint(error.completed.stderr),
        }
    except Exception as error:
        elapsed = time.perf_counter() - started
        return {
            "target": target,
            "status": "failed",
            "stage": stage,
            "elapsed_seconds": round(elapsed, 3),
            "error_type": type(error).__name__,
        }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_junit(path: Path, results: Sequence[Mapping[str, Any]], elapsed: float) -> None:
    failures = sum(result["status"] != "passed" for result in results)
    suite = ET.Element(
        "testsuite",
        {
            "name": "fastplms-python-support",
            "tests": str(len(results)),
            "failures": str(failures),
            "errors": "0",
            "skipped": "0",
            "time": f"{elapsed:.3f}",
        },
    )
    properties = ET.SubElement(suite, "properties")
    ET.SubElement(
        properties,
        "property",
        {"name": "canonical_gpu_python", "value": CANONICAL_GPU_PYTHON},
    )
    for result in results:
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": "tools.remote.python_matrix",
                "name": f"python-{result['target']}",
                "time": f"{float(result['elapsed_seconds']):.3f}",
            },
        )
        if result["status"] != "passed":
            failure = ET.SubElement(
                case,
                "failure",
                {
                    "message": str(result.get("stage", "matrix-member")),
                    "type": "PythonSupportFailure",
                },
            )
            failure.text = json.dumps(result, indent=2, sort_keys=True)
        output = ET.SubElement(case, "system-out")
        output.text = json.dumps(result, sort_keys=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    ET.ElementTree(suite).write(temporary, encoding="utf-8", xml_declaration=True)
    temporary.replace(path)


def run_matrix(
    *,
    project_root: Path,
    output: Path,
    junit_output: Path,
    versions: Sequence[str] = PYTHON_SUPPORT_VERSIONS,
) -> int:
    """Run every support smoke and persist complete machine-readable results."""

    project_root = project_root.resolve()
    required_paths = (
        project_root / "src/fastplms",
        project_root / "requirements/profiles/runtime.in",
        project_root / "requirements/constraints/validation.txt",
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"FastPLMs source workspace is incomplete: {missing}")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required for the Python support matrix.")
    if not versions or len(versions) != len(set(versions)):
        raise ValueError("Python support versions must be a non-empty unique sequence.")

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="fastplms-python-support-") as temporary:
        temporary_root = Path(temporary)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(4, len(versions)),
            thread_name_prefix="fastplms-python-support",
        ) as executor:
            futures = {
                target: executor.submit(
                    _run_member,
                    uv=uv,
                    project_root=project_root,
                    temporary_root=temporary_root,
                    target=target,
                )
                for target in versions
            }
            results = [futures[target].result() for target in versions]

    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": 2,
        "canonical_gpu_python": CANONICAL_GPU_PYTHON,
        "support_matrix": list(versions),
        "elapsed_seconds": round(elapsed, 3),
        "results": results,
    }
    _atomic_json(output, payload)
    _write_junit(junit_output, results, elapsed)
    passed = sum(result["status"] == "passed" for result in results)
    print(f"Python support matrix: {passed}/{len(results)} passed", flush=True)
    print(output, flush=True)
    print(junit_output, flush=True)
    return 0 if passed == len(results) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("artifacts/python-matrix.json"))
    parser.add_argument(
        "--junit-output",
        type=Path,
        default=Path("artifacts/junit/python-matrix.xml"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return run_matrix(
        project_root=arguments.project_root,
        output=arguments.output,
        junit_output=arguments.junit_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
