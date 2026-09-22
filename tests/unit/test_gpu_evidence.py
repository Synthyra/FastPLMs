"""The Modal evidence launcher: fixed stages, the cost cap, and the upload boundary."""

from __future__ import annotations

import json
import pytest

from pathlib import Path

from fastplms.registry import get_model_registry
from tools.confidence.budget import BudgetExceeded, BudgetLedger
from tools.gpu_evidence import config, lever_bench
from tools.gpu_evidence.launch import (
    failing_tests_by_cause,
    junit_counts,
    reservation_dollars,
    reserve_within_cap,
)
from tools.gpu_evidence.record_backend_evidence import backend_evidence
from tools.gpu_evidence.source import (
    ROOT,
    SOURCE_DIRECTORIES,
    SOURCE_FILES,
    excluded_from_upload,
)
from tools.gpu_evidence.stages import STAGES, stage_arguments
from tools.gpu_evidence.typing_check import (
    changed_python_files,
    introduced_errors,
    parse_errors,
)
from tools.remote.run import _RELEASE_LOCAL_PARITY_TESTS


def test_pinned_versions_match_the_validation_constraints() -> None:
    constraints = (ROOT / "requirements/constraints/validation.txt").read_text(encoding="utf-8")

    assert f"torch=={config.TORCH_VERSION}" in constraints.split()
    assert f"transformers=={config.TRANSFORMERS_VERSION}" in constraints.split()


def test_every_uploaded_path_exists_and_none_is_the_repository_root() -> None:
    for directory in SOURCE_DIRECTORIES:
        assert (ROOT / directory).is_dir(), directory
        assert Path(directory).parts, directory
    for file_name in SOURCE_FILES:
        assert (ROOT / file_name).is_file(), file_name


@pytest.mark.parametrize(
    "path",
    (
        ".env",
        ".env.local",
        "prod.env",
        ".secrets.env",
        "nested/.envrc",
        "keys/host.pem",
        "keys/HOST.KEY",
        "bundle.p12",
        "bundle.pfx",
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".git-credentials",
        "aws/credentials",
        "src/__pycache__/module.cpython-312.pyc",
        "tests/.pytest_cache/state",
    ),
)
def test_credential_shaped_and_build_paths_are_never_uploaded(path: str) -> None:
    assert excluded_from_upload(Path(path))


@pytest.mark.parametrize(
    "path",
    (
        "src/fastplms/registry.py",
        "requirements/core.in",
        "docs/environment.md",
        "tests/unit/test_environment.py",
        "kernels.lock",
        ".secrets.env.template",
    ),
)
def test_ordinary_source_is_uploaded(path: str) -> None:
    assert not excluded_from_upload(Path(path))


def test_stage_table_is_well_formed() -> None:
    assert set(STAGES) == {
        "cpu-contract",
        "unit",
        "parity-local",
        "flash-integration",
        "probe",
        "typing",
        "lever-bench",
        "flash-lever-bench",
        "runner-lever-bench",
        "backend-bench",
        "packed-probe",
        "fold-bench",
        "fold-bench-long",
        "fold-peak-memory",
        "fold-bench-smoke",
    }
    for name, spec in STAGES.items():
        assert spec.name == name
        if spec.image == "fold":
            assert spec.device == "gpu"
            worker_timeout = config.FOLD_WORKER_TIMEOUT_SECONDS
        elif spec.device == "cpu":
            worker_timeout = config.CPU_WORKER_TIMEOUT_SECONDS
        else:
            worker_timeout = config.GPU_WORKER_TIMEOUT_SECONDS
        assert 0 < spec.timeout_seconds <= worker_timeout
        for argument in spec.arguments:
            if argument.startswith(("tests/", "tools/")):
                assert (ROOT / argument).exists(), argument


def test_stages_mirror_the_repository_suite_commands() -> None:
    # The Linux-only pre-merge gate documented in docs/testing.md.
    assert STAGES["cpu-contract"].device == "cpu"
    assert STAGES["cpu-contract"].arguments[:5] == (
        "-m",
        "pytest",
        "tests/cpu",
        "-m",
        "cpu_contract",
    )
    assert STAGES["parity-local"].arguments == ("-m", "pytest", *_RELEASE_LOCAL_PARITY_TESTS)


def test_selection_narrows_only_pytest_stages() -> None:
    arguments = stage_arguments(STAGES["unit"], "rotary and not slow", "/tmp/junit.xml")

    assert arguments == (
        "-m",
        "pytest",
        "tests/unit",
        "--ignore-glob=tests/unit/test_confidence_*.py",
        "--junitxml=/tmp/junit.xml",
        "-k",
        "rotary and not slow",
    )
    with pytest.raises(ValueError, match="not a pytest stage"):
        stage_arguments(STAGES["probe"], "anything", "/tmp/junit.xml")
    assert stage_arguments(STAGES["probe"], None, "/tmp/junit.xml") == STAGES["probe"].arguments


@pytest.mark.parametrize("selection", ("a; rm -rf /", "$(whoami)", "x | y", "`id`", "a && b", ""))
def test_selection_rejects_shell_metacharacters(selection: str) -> None:
    with pytest.raises(ValueError, match="may contain only"):
        stage_arguments(STAGES["unit"], selection, "/tmp/junit.xml")


def test_reservation_covers_startup_probe_and_stage_at_the_worker_rate() -> None:
    spec = STAGES["probe"]
    bounded_seconds = (
        config.STARTUP_TIMEOUT_SECONDS
        + config.ENVIRONMENT_PROBE_TIMEOUT_SECONDS
        + spec.timeout_seconds
    )

    assert reservation_dollars(spec, "L4") == pytest.approx(
        bounded_seconds * config.worker_rate("L4")
    )
    assert config.worker_rate(None) < config.worker_rate("L4") < config.worker_rate("H100")
    assert config.DEFAULT_GPU in config.GPU_CHOICES


def test_cap_refuses_a_stage_before_anything_is_reserved(tmp_path: Path) -> None:
    ledger = BudgetLedger(tmp_path / "budget.json")
    spec = STAGES["unit"]
    worst_case = reservation_dollars(spec, "L4")

    with pytest.raises(BudgetExceeded, match="exceeds the"):
        reserve_within_cap(ledger, spec, "L4", max_dollars=worst_case / 2)
    assert ledger.entries == []
    assert not (tmp_path / "budget.json").exists()

    reservation = reserve_within_cap(ledger, spec, "L4", max_dollars=worst_case * 1.5)
    assert ledger.committed() == pytest.approx(worst_case)
    # A second identical stage no longer fits under the same cap.
    with pytest.raises(BudgetExceeded):
        reserve_within_cap(ledger, spec, "L4", max_dollars=worst_case * 1.5)
    # An observed receipt replaces the worst case and frees the remainder.
    ledger.complete(reservation, worst_case / 10)
    reserve_within_cap(ledger, spec, "L4", max_dollars=worst_case * 1.5)


def test_junit_counts_total_every_suite() -> None:
    report = (
        '<testsuites><testsuite tests="3" failures="1" errors="0" skipped="1"/>'
        '<testsuite tests="2" failures="0" errors="1" skipped="0"/></testsuites>'
    )

    assert junit_counts(report) == {"tests": 5, "failures": 1, "errors": 1, "skipped": 1}
    assert junit_counts('<testsuite tests="4" failures="0" errors="0" skipped="0"/>') == {
        "tests": 4,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    assert junit_counts(None) is None


def test_budget_overruns_are_reported_apart_from_assertion_failures() -> None:
    budget = "CPU contract exceeded its 10s budget: 10.02s across setup/call/teardown"
    report = f"""
    <testsuite>
      <testcase classname="tests.cpu.a" name="passes"/>
      <testcase classname="tests.cpu.a" name="slow_worker">
        <failure message="{budget}"/><error message="failed on teardown: {budget}"/>
      </testcase>
      <testcase classname="tests.cpu.a" name="wrong_answer">
        <failure message="assert 1 == 2"/>
      </testcase>
      <testcase classname="tests.cpu.a" name="slow_and_wrong">
        <failure message="assert 1 == 2"/><error message="{budget}"/>
      </testcase>
    </testsuite>
    """

    assert failing_tests_by_cause(report) == {
        # A test that is both slow and wrong is a real failure, never excused.
        "assertion": ["tests.cpu.a::slow_and_wrong", "tests.cpu.a::wrong_answer"],
        "time_budget": ["tests.cpu.a::slow_worker"],
    }
    assert failing_tests_by_cause(None) == {"assertion": [], "time_budget": []}


def test_mypy_errors_are_grouped_by_file_without_line_numbers() -> None:
    output = "\n".join(
        (
            "src/a.py:10: error: Incompatible types  [assignment]",
            "src/a.py:99:5: error: Incompatible types  [assignment]",
            r"src\b.py:3: error: Missing return  [return]",
            "src/a.py:12: note: see the docs",
            "Found 3 errors in 2 files",
        )
    )

    errors = parse_errors(output)

    assert errors["src/a.py"]["Incompatible types  [assignment]"] == 2
    assert errors["src/b.py"]["Missing return  [return]"] == 1
    assert set(errors) == {"src/a.py", "src/b.py"}


def test_only_errors_beyond_the_baseline_count_as_introduced() -> None:
    baseline = parse_errors("src/a.py:1: error: old  [misc]\nsrc/a.py:2: error: old  [misc]")
    shifted = parse_errors("src/a.py:7: error: old  [misc]\nsrc/a.py:9: error: old  [misc]")
    grown = parse_errors(
        "\n".join(
            (
                "src/a.py:1: error: old  [misc]",
                "src/a.py:2: error: old  [misc]",
                "src/a.py:3: error: old  [misc]",
                "src/new.py:1: error: fresh  [misc]",
            )
        )
    )

    # Moving existing errors to other lines introduces nothing.
    assert introduced_errors(shifted, baseline) == {}
    assert introduced_errors(grown, baseline) == {
        "src/a.py": ["old  [misc]"],
        "src/new.py": ["fresh  [misc]"],
    }


def test_changed_files_ignore_line_endings_and_include_new_files(tmp_path: Path) -> None:
    for tree in ("candidate", "baseline"):
        (tmp_path / tree / "src").mkdir(parents=True)
    (tmp_path / "baseline/src/same.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    (tmp_path / "candidate/src/same.py").write_bytes(b"x = 1\ny = 2\n")
    (tmp_path / "baseline/src/edited.py").write_bytes(b"x = 1\n")
    (tmp_path / "candidate/src/edited.py").write_bytes(b"x = 2\n")
    (tmp_path / "candidate/src/added.py").write_bytes(b"x = 3\n")

    changed = changed_python_files(tmp_path / "candidate", tmp_path / "baseline", "src")

    assert changed == ["src/added.py", "src/edited.py"]


def test_bench_workloads_are_wired_to_stages_and_to_the_manifest() -> None:
    registry = get_model_registry()
    manifest_families = {"esm2": "esm2", "esmpp": "esm_plusplus", "dplm": "dplm"}

    for bench_family, backends in lever_bench.MASKED_BACKENDS.items():
        advertised = registry.families[manifest_families[bench_family]].attention
        flash_backends = tuple(name for name in advertised if name.startswith("flash_attention"))
        # SDPA is the reference each FlashAttention backend is compared against.
        assert backends == ("sdpa", *flash_backends)
    for stage in ("flash-lever-bench", "runner-lever-bench", "backend-bench", "lever-bench"):
        arguments = STAGES[stage].arguments
        if "--workloads" in arguments:
            selected = arguments[arguments.index("--workloads") + 1 :]
            assert selected
            assert set(selected) <= set(lever_bench.WORKLOADS)
    masked_names = {
        name
        for name in lever_bench.WORKLOADS
        if any(f"-{kind}-b8-" in name for kind in lever_bench.BATCH_ROW_LENGTHS)
    }
    assert masked_names.isdisjoint(lever_bench.DEFAULT_TREE_WORKLOADS)
    assert all(len(rows) == 8 for rows in lever_bench.BATCH_ROW_LENGTHS.values())


def _write_backend_run(directory: Path, status: str = "passed") -> Path:
    cases = {
        f"{family}-{kind}": {
            "median_ms": {backend: 10.0 for backend in backends},
            "per_round_ms": {backend: [10.0, 10.0, 10.0] for backend in backends},
            "speedup_over_sdpa": {backend: 1.0 for backend in backends},
            "failures": {},
        }
        for family, backends in lever_bench.MASKED_BACKENDS.items()
        for kind in lever_bench.BATCH_ROW_LENGTHS
    }
    directory.mkdir()
    receipt = {
        "stage": "backend-bench",
        "status": status,
        "source": {"revision": "0" * 40, "dirty": True, "changed_paths": 3},
        "result": {"environment": {"gpu": "NVIDIA L4", "torch": config.TORCH_VERSION}},
    }
    (directory / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    (directory / "output.txt").write_text(
        "worker log line\n" + json.dumps(cases, indent=2) + "\n", encoding="utf-8"
    )
    return directory


def test_backend_evidence_copies_numbers_from_passed_runs_only(tmp_path: Path) -> None:
    evidence = backend_evidence([_write_backend_run(tmp_path / "run-a")])

    (run,) = evidence["runs"]
    assert run["run"] == "run-a"
    assert run["environment"]["gpu"] == "NVIDIA L4"
    assert run["source"]["dirty"] is True
    assert set(run["cases"]) == {
        f"{family}-{kind}"
        for family in lever_bench.MASKED_BACKENDS
        for kind in lever_bench.BATCH_ROW_LENGTHS
    }
    assert run["cases"]["esm2-padded"]["median_ms"]["flash_attention_2"] == 10.0
    assert "per_round_ms" not in run["cases"]["esm2-padded"]
    assert evidence["residues_per_batch"] == {"padded": 2336, "full": 4096}

    with pytest.raises(ValueError, match="is not a passed backend-bench run"):
        backend_evidence([_write_backend_run(tmp_path / "run-b", status="failed")])


def test_fold_bench_series_name_pinned_checkpoints_and_isolate_the_official_interpreter() -> None:
    from tools.gpu_evidence import fold_bench

    labels = [series.label for series in fold_bench.SERIES]
    assert len(set(labels)) == len(labels)
    every_series = (
        *fold_bench.SERIES,
        *fold_bench.SMOKE_SERIES,
        *fold_bench.LONG_SERIES,
        *fold_bench.PEAK_SERIES,
    )
    for series in every_series:
        command, environment = fold_bench.worker_command(series, [])
        revision = command[command.index("--revision") + 1]
        assert len(revision) == 40
        if series.implementation == "upstream":
            assert command[0] == fold_bench.REFERENCE_PYTHON
            assert command[command.index("--repo") + 1].startswith("biohub/")
            # The official interpreter must not import the candidate tree.
            assert environment["PYTHONPATH"] == ""
        else:
            assert command[command.index("--repo") + 1].startswith("Synthyra/")
            tree = "/baseline" if series.implementation == "fastplms-baseline" else "/workspace"
            assert environment["PYTHONPATH"] == f"{tree}/src"
        assert ("--experimental" in command) == (series.model_id != "esmfold2")
    # The baseline tree predates the windowed mode, so it may only run dense attention.
    assert all(
        series.atom_attention == "dense" and not series.compare_atom_attention
        for series in every_series
        if series.implementation != "fastplms"
    )


def test_aligned_rmsd_ignores_rigid_motion_and_measures_deformation() -> None:
    import torch

    from tools.gpu_evidence.fold_worker import aligned_rmsd, synthetic_sequence

    generator = torch.Generator().manual_seed(0)
    coordinates = torch.randn(40, 3, generator=generator)  # (residues, xyz)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] = -rotation[:, 0]
    moved = coordinates @ rotation + torch.tensor([3.0, -2.0, 5.0])

    assert aligned_rmsd(coordinates, moved) < 1e-5
    # A mirror image is not a rigid motion, so it must not align to zero.
    assert aligned_rmsd(coordinates, coordinates * torch.tensor([1.0, 1.0, -1.0])) > 0.5
    stretched = coordinates * torch.tensor([1.5, 1.0, 1.0])
    assert aligned_rmsd(coordinates, stretched) > 0.1
    assert synthetic_sequence(64) == synthetic_sequence(64) and len(synthetic_sequence(64)) == 64
