from __future__ import annotations

from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
SCOPE_PATH = ROOT / "tools" / "typing-critical-files.txt"
EXPECTED_CRITICAL_TYPING_SCOPE = (
    "benchmarks",
    "examples/binder_design_fastplms.py",
    "examples/fine_tuning.py",
    "src/fastplms/attention",
    "src/fastplms/embeddings",
    "src/fastplms/registry.py",
    "tools/artifacts",
    "tools/conversion",
    "tools/remote",
    "tools/source_provenance.py",
)


def test_critical_typing_scope_is_explicit_complete_and_portable() -> None:
    entries = tuple(
        line.strip()
        for line in SCOPE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )

    assert entries == EXPECTED_CRITICAL_TYPING_SCOPE
    assert entries == tuple(sorted(set(entries)))
    for entry in entries:
        path = PurePosixPath(entry)
        assert path.as_posix() == entry
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert ROOT.joinpath(*path.parts).exists()


def test_static_ci_consumes_the_checked_typing_scope_without_inline_shrinkage() -> None:
    workflow = (ROOT / ".github" / "workflows" / "cpu-contracts.yml").read_text(
        encoding="utf-8"
    )

    assert "mapfile -t MYPY_TARGETS < tools/typing-critical-files.txt" in workflow
    assert '"${MYPY_TARGETS[@]}"' in workflow
    assert "src/fastplms/registry.py tools/remote" not in workflow
