from __future__ import annotations

import ast
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

from tools.artifacts.generate_docs import synchronize
from tools.debug.check_notation import (
    iter_repository_files,
    scan_repository,
    violations_in_text,
)

ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_ROOTS = (
    ROOT / "README.md",
    ROOT / "THIRD_PARTY_NOTICES.md",
    ROOT / "docs",
    ROOT / "model_cards",
    ROOT / "vendor" / "README.md",
)
FENCE_PATTERN = re.compile(
    r"^```(?P<language>[A-Za-z0-9_+-]*)[^\n]*\n(?P<body>.*?)^```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\((?P<target>[^)]+)\)")
UNBACKED_CLAIM_PATTERNS = (
    re.compile(
        r"\b(?:is|are|has been|have been)\s+"
        r"(?:fully\s+|exactly\s+)?equivalent\b",
        re.I,
    ),
    re.compile(
        r"\b(?:state[- ]of[- ]the[- ]art|outperforms?|"
        r"\d+(?:\.\d+)?\s*[x\u00d7]\s+(?:faster|speedup)|"
        r"\d+(?:\.\d+)?%\s+faster)\b",
        re.I,
    ),
)


def _markdown_files() -> tuple[Path, ...]:
    paths: list[Path] = []
    for candidate in MARKDOWN_ROOTS:
        if candidate.is_file():
            paths.append(candidate)
        elif candidate.is_dir():
            paths.extend(sorted(candidate.rglob("*.md")))
    return tuple(paths)


def _python_snippet(path: Path, marker: str) -> str:
    text = path.read_text(encoding="utf-8")
    matches = [
        match.group("body")
        for match in FENCE_PATTERN.finditer(text)
        if match.group("language").lower() in {"python", "py"} and marker in match.group("body")
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"Expected one Python snippet containing {marker!r} in {path}, found {len(matches)}."
        )
    return matches[0]


def _local_link_target(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    split = urlsplit(target)
    if split.scheme or split.netloc or not split.path:
        return None
    decoded = unquote(split.path)
    destination = ROOT / decoded.lstrip("/") if decoded.startswith("/") else source.parent / decoded
    resolved = destination.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise AssertionError(
            f"Documentation link escapes the repository: {source}: {raw_target}"
        ) from error
    return resolved


def test_shape_notation_detector_rejects_square_and_uppercase_dimensions() -> None:
    text = (
        "H: [" + "B, L, 81, 2560]\n"
        "Z: (" + "B, L, D)\n"
        "M: [" + "n_atoms]\n"
        "X: [" + "samples, atoms, 3]"
    )
    violations = list(violations_in_text(text, path=Path("example.md")))
    assert len(violations) == 4


def test_repository_documentation_uses_canonical_shape_notation() -> None:
    violations = scan_repository(ROOT)
    assert not violations, "\n" + "\n".join(violation.render(ROOT) for violation in violations)


def test_notation_inventory_includes_container_and_provenance_docs() -> None:
    paths = {path.relative_to(ROOT).as_posix() for path in iter_repository_files(ROOT)}
    assert {
        "LICENSES/README.md",
        "THIRD_PARTY_NOTICES.md",
        "docker/Dockerfile",
        "docker/docker-bake.hcl",
        "docker/compose.yaml",
        "vendor/README.md",
    }.issubset(paths)


def test_manifest_generated_documentation_is_current() -> None:
    failures = synchronize(ROOT, check=True)
    assert not failures, "\n" + "\n".join(failures)


def test_documentation_local_links_resolve() -> None:
    failures: list[str] = []
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        for match in LINK_PATTERN.finditer(text):
            target = _local_link_target(path, match.group("target"))
            if target is not None and not target.exists():
                line = text.count("\n", 0, match.start()) + 1
                failures.append(
                    f"{path.relative_to(ROOT)}:{line}: missing link target "
                    f"{target.relative_to(ROOT)}"
                )
    assert not failures, "\n" + "\n".join(failures)


def test_python_documentation_fences_compile() -> None:
    failures: list[str] = []
    count = 0
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        for match in FENCE_PATTERN.finditer(text):
            if match.group("language").lower() not in {"python", "py"}:
                continue
            count += 1
            line = text.count("\n", 0, match.start("body")) + 1
            try:
                ast.parse(match.group("body"), filename=f"{path}:{line}")
            except SyntaxError as error:
                failures.append(f"{path.relative_to(ROOT)}:{line}: {error.msg}")
    assert count > 0, "No executable Python documentation snippets were found."
    assert not failures, "\n" + "\n".join(failures)


def test_readme_embedding_snippet_executes(monkeypatch) -> None:
    import fastplms

    observed: dict[str, object] = {}

    def fake_embed_dataset(model, inputs, **kwargs):
        observed.update(model=model, inputs=inputs, kwargs=kwargs)
        return object()

    monkeypatch.setattr(fastplms, "embed_dataset", fake_embed_dataset)
    namespace = {"model": object()}
    snippet = _python_snippet(ROOT / "README.md", "EmbeddingInput, embed_dataset")
    exec(compile(snippet, "README.md", "exec"), namespace)

    inputs = observed["inputs"]
    assert [record.id for record in inputs] == ["protein-a", "protein-a"]
    assert observed["kwargs"] == {
        "batch_size": 2,
        "pooling": ("mean", "std"),
        "output": "embeddings",
    }


def test_readme_automodel_snippet_executes_without_network(monkeypatch) -> None:
    import transformers

    observed: dict[str, object] = {}

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            observed.update(model_id=model_id, kwargs=kwargs)
            return object()

    monkeypatch.setattr(transformers, "AutoModel", FakeAutoModel)
    snippet = _python_snippet(ROOT / "README.md", 'attn_implementation="flex_attention"')
    exec(compile(snippet, "README.md", "exec"), {})

    assert observed == {
        "model_id": "Synthyra/ESM2-150M",
        "kwargs": {
            "trust_remote_code": True,
            "attn_implementation": "flex_attention",
        },
    }


def test_documentation_does_not_make_unbacked_equivalence_or_speed_claims() -> None:
    failures: list[str] = []
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        for pattern in UNBACKED_CLAIM_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                failures.append(
                    f"{path.relative_to(ROOT)}:{line}: unbacked claim {match.group()!r}"
                )
    assert not failures, "\n" + "\n".join(failures)
