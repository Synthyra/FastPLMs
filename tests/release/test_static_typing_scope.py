from __future__ import annotations

from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
SCOPE_PATH = ROOT / "tools" / "typing-critical-files.txt"
DIAGNOSTIC_SCOPE_PATH = ROOT / "tools" / "typing-diagnostic-files.txt"
EXPECTED_CRITICAL_TYPING_SCOPE = (
    "src/fastplms/registry.py",
    "tools/artifacts/build.py",
    "tools/artifacts/publish.py",
    "tools/conversion/state_transforms.py",
    "tools/source_provenance.py",
)
EXPECTED_DIAGNOSTIC_TYPING_SCOPE = (
    "benchmarks",
    "examples",
    "src/fastplms",
    "tools",
)


def _scope_entries(path: Path) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _assert_portable_scope(entries: tuple[str, ...]) -> None:
    assert entries == tuple(sorted(set(entries)))
    for entry in entries:
        path = PurePosixPath(entry)
        assert path.as_posix() == entry
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert ROOT.joinpath(*path.parts).exists()


def test_critical_typing_scope_is_explicit_complete_and_portable() -> None:
    entries = _scope_entries(SCOPE_PATH)

    assert entries == EXPECTED_CRITICAL_TYPING_SCOPE
    _assert_portable_scope(entries)


def test_diagnostic_typing_scope_retains_the_full_migration_surface() -> None:
    entries = _scope_entries(DIAGNOSTIC_SCOPE_PATH)

    assert entries == EXPECTED_DIAGNOSTIC_TYPING_SCOPE
    _assert_portable_scope(entries)
    discovered = {
        path.relative_to(ROOT).as_posix()
        for target in entries
        for path in ROOT.joinpath(*PurePosixPath(target).parts).rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    }
    covered = {
        path.relative_to(ROOT).as_posix()
        for path in (
            *ROOT.joinpath("benchmarks").rglob("*.py"),
            *ROOT.joinpath("examples").rglob("*.py"),
            *ROOT.joinpath("src", "fastplms").rglob("*.py"),
            *ROOT.joinpath("tools").rglob("*.py"),
        )
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert discovered == covered
    assert not any(
        path.startswith(("tests/", "vendor/", "build/", "dist/", "artifacts/"))
        for path in discovered
    )


def test_manual_validation_documents_the_checked_typing_scope() -> None:
    testing = (ROOT / "docs" / "testing.md").read_text(encoding="utf-8")

    assert "tools/typing-critical-files.txt" in testing
    assert "--explicit-package-bases" in testing
    assert "--follow-imports=silent" in testing
