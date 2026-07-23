"""Deterministic no-regression gate for the broad FastPLMs mypy surface."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = 1
BASELINE_REVISION = "c240d8a85eabcf5f73d7cf2618c4191295f1df5b"
BASELINE_PYTHON_VERSION = "3.12.13"
BASELINE_MYPY_VERSION = "1.20.2"
BASELINE_CHECKED_SOURCE_FILES = 148
BASELINE_ERROR_COUNT = 1064
BASELINE_ERROR_FILE_COUNT = 88
BASELINE_SOURCE_INVENTORY_SHA256 = (
    "f4ec439a48f012353d2f4932dda6c46db09178869b428a6911d1da604d1975bf"
)
BASELINE_SOURCE_TREE_SHA256 = (
    "9732188015f6fa29166428cf9c3320ee7b8045aa9cec52bf154dba671c5e65b3"
)
BASELINE_RAW_REPORT_SHA256 = (
    "fe3e51a917b33c7650943340722258a518eae61c77bd1ebe0efd9384ad1f5a62"
)
BASELINE_FINGERPRINT_SHA256 = (
    "4d2422a34fb52dcffa9d112e7203437c726d76ad8798fb14fea77ad3f6317784"
)
REQUIRED_SCOPE_TARGETS = ("benchmarks", "examples", "src/fastplms", "tools")
MYPY_COMMAND = (
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

_ERROR_PATTERN = re.compile(
    r"^(?P<path>.+?):(?P<line>[0-9]+)(?::(?P<column>[0-9]+))?: "
    r"error: (?P<message>.+?)  \[(?P<code>[^][]+)\]$"
)
_FOUND_PATTERN = re.compile(
    r"^Found (?P<errors>[0-9]+) errors? in (?P<files>[0-9]+) files? "
    r"\(checked (?P<checked>[0-9]+) source files?\)$"
)
_SUCCESS_PATTERN = re.compile(
    r"^Success: no issues found in (?P<checked>[0-9]+) source files?$"
)
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class TypingGateError(ValueError):
    """Raised when typing evidence is incomplete, malformed, or inconsistent."""


@dataclass(frozen=True, order=True)
class Fingerprint:
    """One line-independent mypy finding identity."""

    path: str
    code: str
    message: str


@dataclass(frozen=True)
class MypySnapshot:
    """Parsed mypy errors plus its mandatory terminal summary."""

    fingerprints: Counter[Fingerprint]
    error_count: int
    error_file_count: int
    checked_source_files: int


@dataclass(frozen=True)
class CounterComparison:
    """Multiplicity-aware delta between baseline and candidate errors."""

    retained: Counter[Fingerprint]
    new: Counter[Fingerprint]
    resolved: Counter[Fingerprint]


def load_scope_manifest(path: Path) -> tuple[str, ...]:
    """Read and validate the exact broad source roots."""

    try:
        entries = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except OSError as error:
        raise TypingGateError(f"Unable to read typing scope manifest: {path}") from error
    if entries != REQUIRED_SCOPE_TARGETS:
        raise TypingGateError(
            "Broad typing scope differs from the required roots: "
            + ", ".join(REQUIRED_SCOPE_TARGETS)
        )
    if entries != tuple(sorted(set(entries))):
        raise TypingGateError("Broad typing scope must be sorted and duplicate-free.")
    for entry in entries:
        value = PurePosixPath(entry)
        if value.is_absolute() or value.as_posix() != entry or ".." in value.parts:
            raise TypingGateError(f"Typing scope contains an unsafe path: {entry!r}")
    return entries


def _path_is_scoped(path: PurePosixPath, scope: Sequence[str]) -> bool:
    for target in scope:
        target_parts = PurePosixPath(target).parts
        if path.parts[: len(target_parts)] == target_parts:
            return True
    return False


def normalize_error_path(value: str, scope: Sequence[str]) -> str:
    """Normalize one mypy path to a safe repository-relative POSIX path."""

    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or (path.parts and path.parts[0].endswith(":"))
        or not _path_is_scoped(path, scope)
        or path.suffix != ".py"
    ):
        raise TypingGateError(f"Mypy emitted an out-of-scope path: {value!r}")
    return path.as_posix()


def parse_mypy_output(text: str, scope: Sequence[str]) -> MypySnapshot:
    """Parse standard mypy text output while deliberately ignoring line numbers."""

    fingerprints: Counter[Fingerprint] = Counter()
    lines = [line for line in text.splitlines() if line.strip()]
    summary_matches: list[tuple[int, re.Match[str], bool]] = []
    for index, line in enumerate(lines):
        found_match = _FOUND_PATTERN.fullmatch(line)
        if found_match is not None:
            summary_matches.append((index, found_match, False))
        success_match = _SUCCESS_PATTERN.fullmatch(line)
        if success_match is not None:
            summary_matches.append((index, success_match, True))
    if len(summary_matches) != 1:
        raise TypingGateError("Mypy output must contain exactly one terminal summary.")
    summary_index, summary_match, success = summary_matches[0]
    if summary_index != len(lines) - 1:
        raise TypingGateError("Mypy terminal summary must be the final nonblank line.")
    if success:
        summary = (0, 0, int(summary_match.group("checked")))
    else:
        summary = (
            int(summary_match.group("errors")),
            int(summary_match.group("files")),
            int(summary_match.group("checked")),
        )
    for line in lines[:summary_index]:
        error_match = _ERROR_PATTERN.fullmatch(line)
        if error_match is not None:
            fingerprint = Fingerprint(
                path=normalize_error_path(error_match.group("path"), scope),
                code=error_match.group("code"),
                message=error_match.group("message"),
            )
            fingerprints[fingerprint] += 1
            continue
        if ": error:" in line:
            raise TypingGateError(f"Unrecognized mypy error line: {line!r}")
    error_count, error_file_count, checked_source_files = summary
    parsed_count = sum(fingerprints.values())
    if parsed_count != error_count:
        raise TypingGateError(
            f"Mypy summary reports {error_count} errors but {parsed_count} were parsed."
        )
    parsed_files = len({fingerprint.path for fingerprint in fingerprints})
    if parsed_files != error_file_count:
        raise TypingGateError(
            f"Mypy summary reports {error_file_count} error files but parsed {parsed_files}."
        )
    return MypySnapshot(
        fingerprints=fingerprints,
        error_count=error_count,
        error_file_count=error_file_count,
        checked_source_files=checked_source_files,
    )


def discover_source_files(repo_root: Path, scope: Sequence[str]) -> tuple[str, ...]:
    """Return every Python source under the exact diagnostic roots."""

    root = repo_root.resolve()
    discovered: set[str] = set()
    for target in scope:
        target_path = root.joinpath(*PurePosixPath(target).parts)
        if not target_path.is_dir() or target_path.is_symlink():
            raise TypingGateError(f"Typing scope root is missing or linked: {target_path}")
        for path in target_path.rglob("*"):
            if "__pycache__" in path.parts:
                continue
            if path.is_symlink():
                raise TypingGateError(f"Typing scope contains a symlink: {path}")
            if path.is_file() and path.suffix == ".py":
                discovered.add(path.relative_to(root).as_posix())
    return tuple(sorted(discovered))


def _source_inventory_sha256(source_files: Sequence[str]) -> str:
    inventory_payload = json.dumps(
        list(source_files),
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(inventory_payload).hexdigest()


def source_digests(repo_root: Path, source_files: Sequence[str]) -> tuple[str, str]:
    """Hash the ordered path inventory and the exact source bytes."""

    root = repo_root.resolve()
    tree_records: list[dict[str, str]] = []
    for relative_name in source_files:
        if (
            normalize_error_path(relative_name, REQUIRED_SCOPE_TARGETS)
            != relative_name
        ):
            raise TypingGateError("Source digest inventory contains an unsafe path.")
        path = root.joinpath(*PurePosixPath(relative_name).parts)
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise TypingGateError(f"Unable to hash scoped source: {path}") from error
        tree_records.append(
            {
                "path": relative_name,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    tree_payload = json.dumps(
        tree_records,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        _source_inventory_sha256(source_files),
        hashlib.sha256(tree_payload).hexdigest(),
    )


def verified_git_head(repo_root: Path, scope: Sequence[str]) -> str:
    """Return the clean exact Git HEAD backing baseline source bytes."""

    root = repo_root.resolve()
    command = ["git", "-c", f"safe.directory={root.as_posix()}"]
    try:
        revision = subprocess.run(
            [*command, "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [*command, "status", "--porcelain=v1", "--untracked-files=all", "--", *scope],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise TypingGateError("Unable to verify baseline Git source identity.") from error
    if not _COMMIT_PATTERN.fullmatch(revision):
        raise TypingGateError("Baseline Git HEAD is not a full lowercase commit.")
    if status.strip():
        raise TypingGateError("Baseline Git source scope is not clean.")
    return revision


def verify_git_source_identity(
    repo_root: Path,
    scope: Sequence[str],
    source_files: Sequence[str],
) -> None:
    """Match every discovered Python file byte-for-byte to the Git HEAD tree."""

    root = repo_root.resolve()
    command = ["git", "-c", f"safe.directory={root.as_posix()}"]
    try:
        tree_output = subprocess.run(
            [*command, "ls-tree", "-r", "-z", "HEAD", "--", *scope],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise TypingGateError("Unable to read baseline Git source tree.") from error
    tree_blobs: dict[str, str] = {}
    try:
        records = (record for record in tree_output.split(b"\0") if record)
        for record in records:
            metadata, raw_name = record.split(b"\t", maxsplit=1)
            mode, object_type, object_id = metadata.decode("ascii").split()
            relative_name = raw_name.decode("utf-8")
            if PurePosixPath(relative_name).suffix != ".py":
                continue
            normalized_name = normalize_error_path(relative_name, scope)
            if (
                normalized_name != relative_name
                or object_type != "blob"
                or mode == "120000"
                or normalized_name in tree_blobs
            ):
                raise TypingGateError("Baseline Git tree contains an invalid Python source.")
            tree_blobs[normalized_name] = object_id
    except (UnicodeDecodeError, ValueError) as error:
        raise TypingGateError("Baseline Git source tree output is malformed.") from error
    if tuple(sorted(tree_blobs)) != tuple(source_files):
        raise TypingGateError(
            "Discovered baseline Python inventory differs from its Git HEAD tree."
        )
    try:
        working_hashes = subprocess.run(
            [*command, "hash-object", "--", *source_files],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise TypingGateError("Unable to hash baseline working-tree sources.") from error
    if len(working_hashes) != len(source_files) or any(
        tree_blobs[relative_name] != object_id
        for relative_name, object_id in zip(source_files, working_hashes, strict=True)
    ):
        raise TypingGateError("Baseline source bytes differ from the Git HEAD tree.")


def compare_counters(
    baseline: Counter[Fingerprint],
    candidate: Counter[Fingerprint],
) -> CounterComparison:
    """Compare errors as multisets, retaining multiplicity."""

    return CounterComparison(
        retained=baseline & candidate,
        new=candidate - baseline,
        resolved=baseline - candidate,
    )


def _counter_records(counter: Counter[Fingerprint]) -> list[dict[str, object]]:
    return [
        {
            "path": fingerprint.path,
            "code": fingerprint.code,
            "message": fingerprint.message,
            "count": count,
        }
        for fingerprint, count in sorted(counter.items())
    ]


def _fingerprint_sha256(counter: Counter[Fingerprint]) -> str:
    payload = json.dumps(
        _counter_records(counter),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _run_mypy(repo_root: Path) -> tuple[int, bytes]:
    """Run the immutable command through the current pinned Python interpreter."""

    command = (sys.executable, *MYPY_COMMAND[1:])
    environment = os.environ.copy()
    environment.pop("MYPYPATH", None)
    environment.pop("MYPY_CONFIG_FILE", None)
    environment["NO_COLOR"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root.resolve(),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    except OSError as error:
        raise TypingGateError("Unable to execute the pinned mypy command.") from error
    return completed.returncode, completed.stdout


def _print_report_tail(raw_report: bytes, *, line_count: int = 25) -> None:
    text = raw_report.decode("utf-8", errors="replace")
    print("\n".join(text.splitlines()[-line_count:]))


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TypingGateError(f"Unable to read typing baseline: {path}") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypingGateError("Typing baseline must be a JSON object.")
    return value


def _require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypingGateError(f"{context} must be a non-empty string.")
    return value


def _require_nonnegative_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypingGateError(f"{context} must be a non-negative integer.")
    return value


def _require_string_tuple(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypingGateError(f"{context} must be a string array.")
    return tuple(value)


def _records_counter(
    value: object,
    *,
    scope: Sequence[str],
    source_files: set[str],
) -> Counter[Fingerprint]:
    if not isinstance(value, list):
        raise TypingGateError("Typing baseline fingerprints must be an array.")
    result: Counter[Fingerprint] = Counter()
    for record in value:
        if not isinstance(record, dict) or set(record) != {"path", "code", "message", "count"}:
            raise TypingGateError("Typing baseline contains a malformed fingerprint.")
        raw_path = _require_string(record["path"], "fingerprint path")
        normalized_path = normalize_error_path(raw_path, scope)
        if normalized_path != raw_path or normalized_path not in source_files:
            raise TypingGateError(
                "Typing baseline fingerprint path is not in its source inventory."
            )
        fingerprint = Fingerprint(
            path=normalized_path,
            code=_require_string(record["code"], "fingerprint code"),
            message=_require_string(record["message"], "fingerprint message"),
        )
        count = _require_nonnegative_int(record["count"], "fingerprint count")
        if count == 0 or fingerprint in result:
            raise TypingGateError("Typing baseline fingerprints must be positive and unique.")
        result[fingerprint] = count
    return result


def _validate_pinned_baseline_identity(value: Mapping[str, object]) -> None:
    expected: dict[str, object] = {
        "checked_source_files": BASELINE_CHECKED_SOURCE_FILES,
        "error_count": BASELINE_ERROR_COUNT,
        "error_file_count": BASELINE_ERROR_FILE_COUNT,
        "source_inventory_sha256": BASELINE_SOURCE_INVENTORY_SHA256,
        "source_tree_sha256": BASELINE_SOURCE_TREE_SHA256,
        "raw_report_sha256": BASELINE_RAW_REPORT_SHA256,
        "fingerprint_sha256": BASELINE_FINGERPRINT_SHA256,
    }
    mismatches = [
        field for field, expected_value in expected.items() if value.get(field) != expected_value
    ]
    if mismatches:
        raise TypingGateError(
            "Typing baseline differs from its immutable c240 identity: "
            + ", ".join(mismatches)
        )


def baseline_payload(
    snapshot: MypySnapshot,
    *,
    scope: Sequence[str],
    source_files: Sequence[str],
    revision: str,
    raw_report: bytes,
    source_inventory_sha256: str,
    source_tree_sha256: str,
    mypy_exit_code: int,
) -> dict[str, object]:
    """Build one deterministic checked baseline payload."""

    if not _COMMIT_PATTERN.fullmatch(revision):
        raise TypingGateError("Baseline revision must be a full lowercase Git commit.")
    if tuple(scope) != REQUIRED_SCOPE_TARGETS:
        raise TypingGateError("Baseline scope differs from the required roots.")
    if tuple(source_files) != tuple(sorted(set(source_files))):
        raise TypingGateError("Baseline source inventory must be sorted and unique.")
    if snapshot.checked_source_files != len(source_files):
        raise TypingGateError(
            "Mypy checked-source count differs from the discovered source inventory."
        )
    expected_inventory_sha256 = _source_inventory_sha256(source_files)
    if source_inventory_sha256 != expected_inventory_sha256:
        raise TypingGateError("Baseline source inventory digest is inconsistent.")
    if not _SHA256_PATTERN.fullmatch(source_tree_sha256):
        raise TypingGateError("Baseline source tree digest is invalid.")
    expected_exit_code = 0 if snapshot.error_count == 0 else 1
    if mypy_exit_code != expected_exit_code:
        raise TypingGateError("Baseline mypy exit status contradicts its terminal summary.")
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_revision": revision,
        "environment": {
            "python": platform.python_version(),
            "mypy": importlib.metadata.version("mypy"),
        },
        "mypy_command": list(MYPY_COMMAND),
        "mypy_exit_code": mypy_exit_code,
        "scope_targets": list(scope),
        "checked_source_files": snapshot.checked_source_files,
        "source_files": list(source_files),
        "source_inventory_sha256": source_inventory_sha256,
        "source_tree_sha256": source_tree_sha256,
        "raw_report_sha256": hashlib.sha256(raw_report).hexdigest(),
        "error_count": snapshot.error_count,
        "error_file_count": snapshot.error_file_count,
        "fingerprint_sha256": _fingerprint_sha256(snapshot.fingerprints),
        "fingerprints": _counter_records(snapshot.fingerprints),
    }


def load_baseline(path: Path) -> tuple[dict[str, object], Counter[Fingerprint]]:
    """Load and validate the immutable c240 typing debt ledger."""

    value = _read_json_object(path)
    required = {
        "schema_version",
        "baseline_revision",
        "environment",
        "mypy_command",
        "mypy_exit_code",
        "scope_targets",
        "checked_source_files",
        "source_files",
        "source_inventory_sha256",
        "source_tree_sha256",
        "raw_report_sha256",
        "error_count",
        "error_file_count",
        "fingerprint_sha256",
        "fingerprints",
    }
    if set(value) != required or value.get("schema_version") != SCHEMA_VERSION:
        raise TypingGateError("Typing baseline schema or field inventory is invalid.")
    if value.get("baseline_revision") != BASELINE_REVISION:
        raise TypingGateError("Typing baseline revision differs from the pinned c240 commit.")
    if (
        _require_string_tuple(value.get("scope_targets"), "baseline scope")
        != REQUIRED_SCOPE_TARGETS
    ):
        raise TypingGateError("Typing baseline scope differs from the required roots.")
    if _require_string_tuple(value.get("mypy_command"), "baseline command") != MYPY_COMMAND:
        raise TypingGateError("Typing baseline command differs from the required invocation.")
    source_files = _require_string_tuple(value.get("source_files"), "baseline source files")
    if source_files != tuple(sorted(set(source_files))):
        raise TypingGateError("Typing baseline source inventory is not sorted and unique.")
    for relative_name in source_files:
        if normalize_error_path(relative_name, REQUIRED_SCOPE_TARGETS) != relative_name:
            raise TypingGateError("Typing baseline contains an unsafe source path.")
    checked = _require_nonnegative_int(
        value.get("checked_source_files"),
        "baseline checked-source count",
    )
    if checked != len(source_files):
        raise TypingGateError("Typing baseline checked-source count differs from its inventory.")
    inventory_digest = _require_string(
        value.get("source_inventory_sha256"),
        "baseline source inventory digest",
    )
    if inventory_digest != _source_inventory_sha256(source_files):
        raise TypingGateError("Typing baseline source inventory digest is invalid.")
    fingerprints = _records_counter(
        value.get("fingerprints"),
        scope=REQUIRED_SCOPE_TARGETS,
        source_files=set(source_files),
    )
    if sum(fingerprints.values()) != _require_nonnegative_int(
        value.get("error_count"),
        "baseline error count",
    ):
        raise TypingGateError("Typing baseline error count differs from its fingerprints.")
    if len({item.path for item in fingerprints}) != _require_nonnegative_int(
        value.get("error_file_count"),
        "baseline error-file count",
    ):
        raise TypingGateError("Typing baseline error-file count differs from its fingerprints.")
    mypy_exit_code = _require_nonnegative_int(
        value.get("mypy_exit_code"),
        "baseline mypy exit status",
    )
    expected_exit_code = 0 if not fingerprints else 1
    if mypy_exit_code != expected_exit_code:
        raise TypingGateError("Typing baseline mypy exit status is inconsistent.")
    digest = _require_string(value.get("fingerprint_sha256"), "baseline fingerprint digest")
    if digest != _fingerprint_sha256(fingerprints):
        raise TypingGateError("Typing baseline fingerprint digest is invalid.")
    for field in ("source_tree_sha256", "raw_report_sha256"):
        digest = _require_string(value.get(field), f"baseline {field}")
        if not _SHA256_PATTERN.fullmatch(digest):
            raise TypingGateError(f"Typing baseline {field} is not a SHA-256 digest.")
    environment = value.get("environment")
    if (
        not isinstance(environment, dict)
        or set(environment) != {"python", "mypy"}
        or environment.get("python") != BASELINE_PYTHON_VERSION
        or environment.get("mypy") != BASELINE_MYPY_VERSION
    ):
        raise TypingGateError("Typing baseline environment identity is invalid.")
    _validate_pinned_baseline_identity(value)
    return value, fingerprints


def compare_payload(
    *,
    baseline: Mapping[str, object],
    baseline_fingerprints: Counter[Fingerprint],
    candidate: MypySnapshot,
    scope: Sequence[str],
    source_files: Sequence[str],
    source_inventory_sha256: str,
    source_tree_sha256: str,
    mypy_exit_code: int,
) -> dict[str, object]:
    """Create the reader-facing candidate comparison and fail-closed reasons."""

    comparison = compare_counters(baseline_fingerprints, candidate.fingerprints)
    reasons: list[str] = []
    if tuple(scope) != REQUIRED_SCOPE_TARGETS:
        reasons.append("diagnostic scope differs from the required roots")
    if tuple(source_files) != tuple(sorted(set(source_files))):
        reasons.append("candidate source inventory is not sorted and unique")
    source_file_set = set(source_files)
    for relative_name in source_files:
        try:
            normalized_name = normalize_error_path(relative_name, scope)
        except TypingGateError:
            reasons.append("candidate source inventory contains an unsafe path")
            break
        if normalized_name != relative_name:
            reasons.append("candidate source inventory contains a non-canonical path")
            break
    fingerprints_outside_inventory = sorted(
        {
            fingerprint.path
            for fingerprint in candidate.fingerprints
            if fingerprint.path not in source_file_set
        }
    )
    if fingerprints_outside_inventory:
        reasons.append("candidate fingerprints fall outside its source inventory")
    if mypy_exit_code not in {0, 1}:
        reasons.append(f"mypy exited with infrastructure status {mypy_exit_code}")
    expected_exit_code = 0 if candidate.error_count == 0 else 1
    if mypy_exit_code in {0, 1} and mypy_exit_code != expected_exit_code:
        reasons.append("mypy exit status contradicts its terminal summary")
    if candidate.checked_source_files != len(source_files):
        reasons.append("mypy checked-source count differs from candidate inventory")
    baseline_files = set(
        _require_string_tuple(baseline.get("source_files"), "baseline source files")
    )
    missing_baseline_files = sorted(baseline_files.difference(source_files))
    if source_inventory_sha256 != _source_inventory_sha256(source_files):
        reasons.append("candidate source inventory digest is inconsistent")
    if not _SHA256_PATTERN.fullmatch(source_tree_sha256):
        reasons.append("candidate source tree digest is invalid")
    if missing_baseline_files:
        reasons.append("candidate source scope removed baseline files")
    if comparison.new:
        reasons.append("candidate introduced typing fingerprints beyond baseline debt")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed" if reasons else "passed",
        "failure_reasons": reasons,
        "baseline_revision": baseline["baseline_revision"],
        "environment": {
            "python": platform.python_version(),
            "mypy": importlib.metadata.version("mypy"),
        },
        "mypy_command": list(MYPY_COMMAND),
        "scope_targets": list(scope),
        "mypy_exit_code": mypy_exit_code,
        "baseline_checked_source_files": baseline["checked_source_files"],
        "candidate_checked_source_files": candidate.checked_source_files,
        "candidate_source_files": list(source_files),
        "candidate_source_inventory_sha256": source_inventory_sha256,
        "candidate_source_tree_sha256": source_tree_sha256,
        "missing_baseline_source_files": missing_baseline_files,
        "fingerprint_paths_outside_candidate_inventory": (
            fingerprints_outside_inventory
        ),
        "baseline_error_count": sum(baseline_fingerprints.values()),
        "candidate_error_count": candidate.error_count,
        "retained_error_count": sum(comparison.retained.values()),
        "new_error_count": sum(comparison.new.values()),
        "resolved_error_count": sum(comparison.resolved.values()),
        "new_fingerprints": _counter_records(comparison.new),
        "resolved_fingerprints": _counter_records(comparison.resolved),
        "candidate_fingerprint_sha256": _fingerprint_sha256(candidate.fingerprints),
    }


def _failed_compare_payload(
    *,
    baseline: Mapping[str, object],
    scope: Sequence[str],
    source_files: Sequence[str],
    source_inventory_sha256: str,
    source_tree_sha256: str,
    mypy_exit_code: int,
    raw_report: bytes,
    reasons: Sequence[str],
) -> dict[str, object]:
    """Create a durable report even when mypy output cannot be parsed."""

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "failure_reasons": list(reasons),
        "baseline_revision": baseline["baseline_revision"],
        "environment": {
            "python": platform.python_version(),
            "mypy": importlib.metadata.version("mypy"),
        },
        "mypy_command": list(MYPY_COMMAND),
        "scope_targets": list(scope),
        "mypy_exit_code": mypy_exit_code,
        "candidate_source_files": list(source_files),
        "candidate_source_inventory_sha256": source_inventory_sha256,
        "candidate_source_tree_sha256": source_tree_sha256,
        "raw_report_sha256": hashlib.sha256(raw_report).hexdigest(),
        "candidate_output_parsed": False,
    }


def _baseline_command(arguments: argparse.Namespace) -> int:
    scope = load_scope_manifest(arguments.scope_manifest)
    environment = {
        "python": platform.python_version(),
        "mypy": importlib.metadata.version("mypy"),
    }
    if environment != {
        "python": BASELINE_PYTHON_VERSION,
        "mypy": BASELINE_MYPY_VERSION,
    }:
        raise TypingGateError("Baseline generation requires Python 3.12.13 and mypy 1.20.2.")
    revision = verified_git_head(arguments.repo_root, scope)
    if revision != BASELINE_REVISION:
        raise TypingGateError("Baseline Git HEAD differs from the pinned c240 commit.")
    source_files = discover_source_files(arguments.repo_root, scope)
    verify_git_source_identity(arguments.repo_root, scope, source_files)
    source_inventory_sha256, source_tree_sha256 = source_digests(
        arguments.repo_root,
        source_files,
    )
    mypy_exit_code, raw_report = _run_mypy(arguments.repo_root)
    _write_bytes(arguments.raw_output, raw_report)
    _print_report_tail(raw_report)
    if verified_git_head(arguments.repo_root, scope) != revision:
        raise TypingGateError("Baseline Git HEAD changed while evidence was generated.")
    final_source_files = discover_source_files(arguments.repo_root, scope)
    verify_git_source_identity(arguments.repo_root, scope, final_source_files)
    final_digests = source_digests(arguments.repo_root, final_source_files)
    if final_source_files != source_files or final_digests != (
        source_inventory_sha256,
        source_tree_sha256,
    ):
        raise TypingGateError("Baseline source tree changed during mypy execution.")
    try:
        text = raw_report.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TypingGateError("Baseline mypy output is not UTF-8.") from error
    snapshot = parse_mypy_output(text, scope)
    payload = baseline_payload(
        snapshot,
        scope=scope,
        source_files=source_files,
        revision=revision,
        raw_report=raw_report,
        source_inventory_sha256=source_inventory_sha256,
        source_tree_sha256=source_tree_sha256,
        mypy_exit_code=mypy_exit_code,
    )
    _validate_pinned_baseline_identity(payload)
    _write_json(arguments.output, payload)
    return 0


def _compare_command(arguments: argparse.Namespace) -> int:
    baseline, baseline_fingerprints = load_baseline(arguments.baseline)
    environment = {
        "python": platform.python_version(),
        "mypy": importlib.metadata.version("mypy"),
    }
    if environment != baseline["environment"]:
        raise TypingGateError("Candidate typing environment differs from the baseline.")
    scope = load_scope_manifest(arguments.scope_manifest)
    source_files = discover_source_files(arguments.repo_root, scope)
    source_inventory_sha256, source_tree_sha256 = source_digests(
        arguments.repo_root,
        source_files,
    )
    mypy_exit_code, raw_report = _run_mypy(arguments.repo_root)
    _write_bytes(arguments.raw_output, raw_report)
    _print_report_tail(raw_report)
    final_source_files = discover_source_files(arguments.repo_root, scope)
    final_inventory_sha256, final_tree_sha256 = source_digests(
        arguments.repo_root,
        final_source_files,
    )
    if final_source_files != source_files or (
        final_inventory_sha256,
        final_tree_sha256,
    ) != (source_inventory_sha256, source_tree_sha256):
        payload = _failed_compare_payload(
            baseline=baseline,
            scope=scope,
            source_files=final_source_files,
            source_inventory_sha256=final_inventory_sha256,
            source_tree_sha256=final_tree_sha256,
            mypy_exit_code=mypy_exit_code,
            raw_report=raw_report,
            reasons=["candidate source tree changed during mypy execution"],
        )
        _write_json(arguments.output, payload)
        return 1
    try:
        text = raw_report.decode("utf-8")
    except UnicodeDecodeError:
        text = ""
    parse_error: TypingGateError | None = None
    try:
        snapshot = parse_mypy_output(text, scope)
    except TypingGateError as error:
        parse_error = error
    if parse_error is not None:
        reasons = []
        if mypy_exit_code not in {0, 1}:
            reasons.append(
                f"mypy exited with infrastructure status {mypy_exit_code}"
            )
        reasons.append(f"candidate mypy output is invalid: {parse_error}")
        payload = _failed_compare_payload(
            baseline=baseline,
            scope=scope,
            source_files=source_files,
            source_inventory_sha256=source_inventory_sha256,
            source_tree_sha256=source_tree_sha256,
            mypy_exit_code=mypy_exit_code,
            raw_report=raw_report,
            reasons=reasons,
        )
        _write_json(arguments.output, payload)
        return 1
    payload = compare_payload(
        baseline=baseline,
        baseline_fingerprints=baseline_fingerprints,
        candidate=snapshot,
        scope=scope,
        source_files=source_files,
        source_inventory_sha256=source_inventory_sha256,
        source_tree_sha256=source_tree_sha256,
        mypy_exit_code=mypy_exit_code,
    )
    payload["raw_report_sha256"] = hashlib.sha256(raw_report).hexdigest()
    payload["candidate_output_parsed"] = True
    _write_json(arguments.output, payload)
    return 0 if payload["status"] == "passed" else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the deterministic baseline/comparison CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--raw-output", type=Path, required=True)
    baseline.add_argument("--scope-manifest", type=Path, required=True)
    baseline.add_argument("--repo-root", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.set_defaults(handler=_baseline_command)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--raw-output", type=Path, required=True)
    compare.add_argument("--scope-manifest", type=Path, required=True)
    compare.add_argument("--repo-root", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(handler=_compare_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run baseline generation or candidate comparison."""

    arguments = build_parser().parse_args(argv)
    try:
        handler = arguments.handler
        if not callable(handler):
            raise TypingGateError("Typing gate command handler is invalid.")
        return int(handler(arguments))
    except (OSError, TypingGateError) as error:
        print(f"typing gate failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
