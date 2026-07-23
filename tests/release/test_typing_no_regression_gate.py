from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pytest

from tools import typing_gate
from tools.typing_gate import (
    BASELINE_CHECKED_SOURCE_FILES,
    BASELINE_ERROR_COUNT,
    BASELINE_ERROR_FILE_COUNT,
    BASELINE_FINGERPRINT_SHA256,
    BASELINE_MYPY_VERSION,
    BASELINE_PYTHON_VERSION,
    BASELINE_RAW_REPORT_SHA256,
    BASELINE_REVISION,
    BASELINE_SOURCE_INVENTORY_SHA256,
    BASELINE_SOURCE_TREE_SHA256,
    MYPY_COMMAND,
    REQUIRED_SCOPE_TARGETS,
    Fingerprint,
    TypingGateError,
    baseline_payload,
    compare_payload,
    discover_source_files,
    load_baseline,
    load_scope_manifest,
    main,
    parse_mypy_output,
)

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tools" / "typing-baselines" / "c240d8a.json"
SCOPE = ROOT / "tools" / "typing-diagnostic-files.txt"


def _error(path: str, line: int, message: str, code: str) -> str:
    return f"{path}:{line}: error: {message}  [{code}]"


def _snapshot(*errors: str, checked: int = 4) -> str:
    files = len({line.split(":", maxsplit=1)[0] for line in errors})
    return "\n".join(
        (
            *errors,
            f"Found {len(errors)} errors in {files} files "
            f"(checked {checked} source files)",
        )
    )


def _success(*, checked: int) -> str:
    return f"Success: no issues found in {checked} source files"


def _inventory_digest(source_files: tuple[str, ...]) -> str:
    payload = json.dumps(list(source_files), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _compare(
    *,
    baseline_fingerprints: Counter[Fingerprint],
    candidate_text: str,
    source_files: tuple[str, ...],
    baseline_source_files: tuple[str, ...] | None = None,
    exit_code: int = 1,
) -> dict[str, object]:
    candidate = parse_mypy_output(candidate_text, REQUIRED_SCOPE_TARGETS)
    baseline_files = baseline_source_files or source_files
    return compare_payload(
        baseline={
            "baseline_revision": BASELINE_REVISION,
            "checked_source_files": len(baseline_files),
            "source_files": list(baseline_files),
        },
        baseline_fingerprints=baseline_fingerprints,
        candidate=candidate,
        scope=REQUIRED_SCOPE_TARGETS,
        source_files=source_files,
        source_inventory_sha256=_inventory_digest(source_files),
        source_tree_sha256="a" * 64,
        mypy_exit_code=exit_code,
    )


def _write_mutated_baseline(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
) -> Path:
    value = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    mutation(value)
    path = tmp_path / "forged-baseline.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_parser_normalizes_paths_and_preserves_multiplicity() -> None:
    text = "\n".join(
        (
            "src\\fastplms\\model.py:12:4: error: Invalid value  [arg-type]",
            "src/fastplms/model.py:999: error: Invalid value  [arg-type]",
            "Found 2 errors in 1 file (checked 4 source files)",
        )
    )

    parsed = parse_mypy_output(text, REQUIRED_SCOPE_TARGETS)

    assert parsed.fingerprints == Counter(
        {Fingerprint("src/fastplms/model.py", "arg-type", "Invalid value"): 2}
    )


def test_parser_requires_exactly_one_final_terminal_summary() -> None:
    duplicate = "\n".join(
        (
            _success(checked=1),
            _success(checked=1),
        )
    )
    nonterminal = "\n".join(
        (
            _success(checked=1),
            "tools/a.py: note: late output",
        )
    )

    with pytest.raises(TypingGateError, match="exactly one"):
        parse_mypy_output(duplicate, REQUIRED_SCOPE_TARGETS)
    with pytest.raises(TypingGateError, match="final nonblank"):
        parse_mypy_output(nonterminal, REQUIRED_SCOPE_TARGETS)


def test_parser_rejects_unrecognized_or_out_of_scope_errors() -> None:
    with pytest.raises(TypingGateError, match="Unrecognized"):
        parse_mypy_output(
            "tools/a.py:1: error: Missing code\n"
            "Found 1 error in 1 file (checked 1 source file)",
            REQUIRED_SCOPE_TARGETS,
        )
    with pytest.raises(TypingGateError, match="out-of-scope"):
        parse_mypy_output(
            _snapshot(_error("tests/test_a.py", 1, "Bad", "arg-type"), checked=1),
            REQUIRED_SCOPE_TARGETS,
        )


def test_payload_fails_for_one_additional_duplicate_error() -> None:
    fingerprint = Fingerprint("benchmarks/run.py", "arg-type", "Bad argument")
    payload = _compare(
        baseline_fingerprints=Counter({fingerprint: 1}),
        candidate_text=_snapshot(
            _error(fingerprint.path, 1, fingerprint.message, fingerprint.code),
            _error(fingerprint.path, 2, fingerprint.message, fingerprint.code),
            checked=1,
        ),
        source_files=(fingerprint.path,),
    )

    assert payload["status"] == "failed"
    assert payload["new_error_count"] == 1
    assert payload["retained_error_count"] == 1


def test_payload_allows_resolved_baseline_errors() -> None:
    retained = Fingerprint("tools/a.py", "arg-type", "Retained")
    resolved = Fingerprint("tools/b.py", "assignment", "Resolved")
    payload = _compare(
        baseline_fingerprints=Counter({retained: 1, resolved: 2}),
        candidate_text=_snapshot(
            _error(retained.path, 9, retained.message, retained.code),
            checked=2,
        ),
        source_files=("tools/a.py", "tools/b.py"),
    )

    assert payload["status"] == "passed"
    assert payload["new_error_count"] == 0
    assert payload["resolved_error_count"] == 2


def test_payload_fails_closed_on_scope_shrinkage() -> None:
    debt = Fingerprint("tools/a.py", "arg-type", "Debt")
    payload = _compare(
        baseline_fingerprints=Counter({debt: 1}),
        candidate_text=_snapshot(
            _error(debt.path, 4, debt.message, debt.code),
            checked=1,
        ),
        source_files=("tools/a.py",),
        baseline_source_files=("tools/a.py", "tools/b.py"),
    )

    assert payload["status"] == "failed"
    assert payload["missing_baseline_source_files"] == ["tools/b.py"]


def test_payload_fails_closed_on_infrastructure_exit() -> None:
    payload = _compare(
        baseline_fingerprints=Counter(),
        candidate_text=_success(checked=1),
        source_files=("tools/a.py",),
        exit_code=2,
    )

    assert payload["status"] == "failed"
    assert payload["failure_reasons"] == [
        "mypy exited with infrastructure status 2"
    ]


def test_payload_rejects_fingerprint_outside_candidate_inventory() -> None:
    fingerprint = Fingerprint("tools/b.py", "arg-type", "Bad")
    payload = _compare(
        baseline_fingerprints=Counter(),
        candidate_text=_snapshot(
            _error(fingerprint.path, 1, fingerprint.message, fingerprint.code),
            checked=1,
        ),
        source_files=("tools/a.py",),
    )

    assert payload["status"] == "failed"
    assert payload["fingerprint_paths_outside_candidate_inventory"] == [
        "tools/b.py"
    ]


def test_baseline_payload_rejects_infrastructure_exit() -> None:
    snapshot = parse_mypy_output(
        _snapshot(_error("tools/a.py", 1, "Debt", "arg-type"), checked=1),
        REQUIRED_SCOPE_TARGETS,
    )
    with pytest.raises(TypingGateError, match="exit status"):
        baseline_payload(
            snapshot,
            scope=REQUIRED_SCOPE_TARGETS,
            source_files=("tools/a.py",),
            revision=BASELINE_REVISION,
            raw_report=b"report",
            source_inventory_sha256=_inventory_digest(("tools/a.py",)),
            source_tree_sha256="a" * 64,
            mypy_exit_code=2,
        )


def test_load_baseline_rejects_forged_out_of_scope_source(tmp_path: Path) -> None:
    def mutate(value: dict[str, object]) -> None:
        source_files = list(value["source_files"])
        source_files.append("tests/forged.py")
        value["source_files"] = sorted(source_files)
        value["checked_source_files"] = len(source_files)

    path = _write_mutated_baseline(tmp_path, mutate)

    with pytest.raises(TypingGateError, match="out-of-scope"):
        load_baseline(path)


def test_load_baseline_rejects_fingerprint_absent_from_inventory(
    tmp_path: Path,
) -> None:
    def mutate(value: dict[str, object]) -> None:
        fingerprints = list(value["fingerprints"])
        fingerprints[0] = {**fingerprints[0], "path": "tools/forged.py"}
        value["fingerprints"] = fingerprints

    path = _write_mutated_baseline(tmp_path, mutate)

    with pytest.raises(TypingGateError, match="not in its source inventory"):
        load_baseline(path)


def test_load_baseline_rejects_inventory_digest_mutation(tmp_path: Path) -> None:
    def mutate(value: dict[str, object]) -> None:
        source_files = list(value["source_files"])
        source_files.append("tools/forged_inventory.py")
        value["source_files"] = sorted(source_files)
        value["checked_source_files"] = len(source_files)

    path = _write_mutated_baseline(tmp_path, mutate)

    with pytest.raises(TypingGateError, match="inventory digest"):
        load_baseline(path)


@pytest.mark.parametrize("field", ["python", "mypy"])
def test_load_baseline_rejects_environment_mutation(
    tmp_path: Path,
    field: str,
) -> None:
    def mutate(value: dict[str, object]) -> None:
        environment = dict(value["environment"])
        environment[field] = "forged"
        value["environment"] = environment

    path = _write_mutated_baseline(tmp_path, mutate)

    with pytest.raises(TypingGateError, match="environment"):
        load_baseline(path)


def test_load_baseline_rejects_environment_field_addition(tmp_path: Path) -> None:
    def mutate(value: dict[str, object]) -> None:
        environment = dict(value["environment"])
        environment["extra"] = "forged"
        value["environment"] = environment

    path = _write_mutated_baseline(tmp_path, mutate)

    with pytest.raises(TypingGateError, match="environment"):
        load_baseline(path)


def test_load_baseline_rejects_command_mutation(tmp_path: Path) -> None:
    def mutate(value: dict[str, object]) -> None:
        command = list(value["mypy_command"])
        command.remove("--strict")
        value["mypy_command"] = command

    path = _write_mutated_baseline(tmp_path, mutate)

    with pytest.raises(TypingGateError, match="command"):
        load_baseline(path)


def test_load_baseline_rejects_self_consistent_forged_debt(
    tmp_path: Path,
) -> None:
    def mutate(value: dict[str, object]) -> None:
        records = [dict(record) for record in value["fingerprints"]]
        records[0]["message"] = f"{records[0]['message']} forged"
        value["fingerprints"] = records
        counter = Counter(
            {
                Fingerprint(
                    path=str(record["path"]),
                    code=str(record["code"]),
                    message=str(record["message"]),
                ): int(record["count"])
                for record in records
            }
        )
        value["fingerprint_sha256"] = typing_gate._fingerprint_sha256(counter)

    path = _write_mutated_baseline(tmp_path, mutate)

    with pytest.raises(TypingGateError, match="immutable c240 identity"):
        load_baseline(path)


@pytest.mark.parametrize("field", ["source_tree_sha256", "raw_report_sha256"])
def test_load_baseline_rejects_valid_but_untrusted_attestation_digest(
    tmp_path: Path,
    field: str,
) -> None:
    def mutate(value: dict[str, object]) -> None:
        value[field] = "0" * 64

    path = _write_mutated_baseline(tmp_path, mutate)

    with pytest.raises(TypingGateError, match="immutable c240 identity"):
        load_baseline(path)


def test_compare_cli_writes_report_for_unparseable_infrastructure_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_output = tmp_path / "mypy.txt"
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        typing_gate,
        "_run_mypy",
        lambda repo_root: (2, b"mypy: error: internal failure\n"),
    )

    exit_code = main(
        (
            "compare",
            "--baseline",
            str(BASELINE),
            "--raw-output",
            str(raw_output),
            "--scope-manifest",
            str(SCOPE),
            "--repo-root",
            str(ROOT),
            "--output",
            str(report),
        )
    )

    assert exit_code == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["candidate_output_parsed"] is False
    assert "infrastructure status 2" in payload["failure_reasons"][0]


def test_baseline_cli_executes_the_owned_mypy_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for relative_name in (
        "benchmarks/a.py",
        "examples/a.py",
        "src/fastplms/a.py",
        "tools/a.py",
    ):
        path = tmp_path / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    scope_manifest = tmp_path / "scope.txt"
    scope_manifest.write_text("\n".join(REQUIRED_SCOPE_TARGETS) + "\n", encoding="utf-8")
    raw_output = tmp_path / "raw.txt"
    output = tmp_path / "baseline.json"
    calls: list[Path] = []

    monkeypatch.setattr(
        typing_gate,
        "verified_git_head",
        lambda repo_root, scope: BASELINE_REVISION,
    )
    monkeypatch.setattr(
        typing_gate,
        "verify_git_source_identity",
        lambda repo_root, scope, source_files: None,
    )
    monkeypatch.setattr(
        typing_gate,
        "_validate_pinned_baseline_identity",
        lambda value: None,
    )

    def fake_run(repo_root: Path) -> tuple[int, bytes]:
        calls.append(repo_root)
        return (
            1,
            _snapshot(
                _error("tools/a.py", 1, "Baseline debt", "arg-type"),
                checked=4,
            ).encode(),
        )

    monkeypatch.setattr(typing_gate, "_run_mypy", fake_run)

    exit_code = main(
        (
            "baseline",
            "--raw-output",
            str(raw_output),
            "--scope-manifest",
            str(scope_manifest),
            "--repo-root",
            str(tmp_path),
            "--output",
            str(output),
        )
    )

    assert exit_code == 0
    assert calls == [tmp_path]
    assert raw_output.read_bytes().endswith(b"(checked 4 source files)")
    assert json.loads(output.read_text(encoding="utf-8"))["mypy_exit_code"] == 1


def test_checked_baseline_manifest_and_command_are_complete() -> None:
    scope = load_scope_manifest(SCOPE)
    baseline, fingerprints = load_baseline(BASELINE)
    source_files = discover_source_files(ROOT, scope)

    assert scope == REQUIRED_SCOPE_TARGETS
    assert MYPY_COMMAND == (
        "python",
        "-m",
        "mypy",
        "--config-file=/dev/null",
        "--python-version",
        "3.12",
        "--strict",
        "--warn-unreachable",
        "--ignore-missing-imports",
        "--no-site-packages",
        "--no-incremental",
        "--explicit-package-bases",
        "--follow-imports=silent",
        "--show-error-codes",
        "--no-color-output",
        "--no-pretty",
        "benchmarks",
        "examples",
        "src/fastplms",
        "tools",
    )
    assert baseline["baseline_revision"] == BASELINE_REVISION
    assert baseline["environment"] == {
        "python": BASELINE_PYTHON_VERSION,
        "mypy": BASELINE_MYPY_VERSION,
    }
    assert baseline["mypy_command"] == list(MYPY_COMMAND)
    assert baseline["mypy_exit_code"] == 1
    assert baseline["checked_source_files"] == BASELINE_CHECKED_SOURCE_FILES
    assert baseline["error_count"] == BASELINE_ERROR_COUNT
    assert baseline["error_file_count"] == BASELINE_ERROR_FILE_COUNT
    assert baseline["error_count"] == sum(fingerprints.values())
    assert (
        baseline["source_inventory_sha256"]
        == BASELINE_SOURCE_INVENTORY_SHA256
    )
    assert baseline["source_tree_sha256"] == BASELINE_SOURCE_TREE_SHA256
    assert baseline["raw_report_sha256"] == BASELINE_RAW_REPORT_SHA256
    assert baseline["fingerprint_sha256"] == BASELINE_FINGERPRINT_SHA256
    assert set(baseline["source_files"]).issubset(source_files)
