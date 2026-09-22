"""Run independent CPU verification batches without mixing pytest conftest policies."""

from __future__ import annotations

import platform
import subprocess
import sys
import time

from importlib.metadata import version
from pathlib import Path


UNIT_PATTERNS = (
    "test_confidence_*.py",
    "test_embeddings_api.py",
    "test_remote_runner_contract.py",
    "test_gpu_evidence.py",
    "test_evidence_store.py",
    "test_esm_rotary.py",
    "test_prepare_esmfold2_small.py",
    "test_build_all_artifacts.py",
    "test_execution*.py",
    "test_verification_cpu.py",
    "test_registry.py",
    "test_import_hygiene.py",
    "test_*rotary*.py",
    "test_attention_interfaces.py",
    "test_fine_tuning_example.py",
    "test_binder_example_contracts.py",
    "test_benchmark*.py",
    "test_esmfold2_leaf_contracts.py",
    "test_esmfold2_reimplemented_leaves.py",
    "test_esmfold2_classification.py",
    "test_esmfold2_public_contracts.py",
    "test_esmfold2_progress.py",
    "test_esmfold2_triangle_chunks.py",
    "test_esmfold2_atom_attention.py",
    "test_boltz_checkpoint_io.py",
    "test_boltz_opm_diagnostic.py",
    "test_boltz_progress.py",
)
RELEASE_FILES = (
    "test_doc_generation.py",
    "test_documentation.py",
    "test_model_card_licenses.py",
    "test_esmc_report_ingestion.py",
    "test_artifacts.py",
    "test_dependency_contracts.py",
    "test_production_source_boundary.py",
    "test_publish_files_only.py",
    "test_binder_example_contracts.py",
)
CPU_FILES = (
    "test_embedding_contracts.py",
    "test_documentation_contracts.py",
    "test_structure_contracts.py",
)


def test_batches(root: Path) -> dict[str, list[str]]:
    """Keep the CPU tier's global wall-clock fixture out of other test collections."""
    units = sorted(
        {
            path.relative_to(root).as_posix()
            for pattern in UNIT_PATTERNS
            for path in (root / "tests/unit").glob(pattern)
        }
    )
    return {
        "unit-integration": [
            *units,
            "tests/integration/test_source_archive_record.py",
            "tests/integration/test_esm3.py::test_esm3_saved_model_loads_without_installed_fastplms",
        ],
        "release": [f"tests/release/{name}" for name in RELEASE_FILES],
        "cpu-contracts": [f"tests/cpu/{name}" for name in CPU_FILES],
    }


def run_batches(root: Path, output: Path, maximum_seconds: float = 1050) -> dict[str, object]:
    """Preserve every command's output and JUnit, including failed or timed-out batches."""
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    batches = []
    for name, paths in test_batches(root).items():
        junit = output / f"{name}.xml"
        command = [
            sys.executable, "-m", "pytest", "-q", "-m", "not gpu", *paths, f"--junitxml={junit}"
        ]
        remaining = maximum_seconds - (time.monotonic() - started)
        batch_started = time.monotonic()
        if remaining <= 0:
            batches.append(
                {"name": name, "command": command, "status": "not_run", "exit_code": None}
            )
            continue
        try:
            result = subprocess.run(
                command, cwd=root, capture_output=True, text=True, timeout=remaining
            )
            stdout, stderr = result.stdout, result.stderr
            code = result.returncode
            status = "passed" if code == 0 else "failed"
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or b""
            stderr = error.stderr or b""
            stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
            stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
            code, status = None, "timed_out"
        batches.append(
            {
                "name": name,
                "command": command,
                "status": status,
                "exit_code": code,
                "elapsed_seconds": time.monotonic() - batch_started,
                "stdout": stdout,
                "stderr": stderr,
                "junit_xml": junit.read_text(encoding="utf-8") if junit.exists() else None,
            }
        )
    return {
        "status": "passed" if all(batch["status"] == "passed" for batch in batches) else "failed",
        "scope": "Selected unit, integration, release, and CPU contracts.",
        "elapsed_seconds": time.monotonic() - started,
        "batches": batches,
    }


def verify() -> dict[str, object]:
    """Collect the installed environment beside the selected contract results."""
    report = run_batches(Path("/workspace"), Path("/tmp/verification"))
    report["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in (
                "torch",
                "transformers",
                "numpy",
                "scipy",
                "pytest",
                "pyarrow",
                "huggingface-hub",
            )
        },
    }
    return report
