"""Plan and synchronize deterministic generated documentation outputs."""

from __future__ import annotations

import os
from pathlib import Path

from fastplms.registry import ModelRegistry, get_model_registry
from tools.artifacts.doc_generation.card_metadata import (
    GENERATED_MARKER,
)
from tools.artifacts.doc_generation.esmc_evidence import (
    EsmcReportSet,
    load_esmc_report_set,
)
from tools.artifacts.doc_generation.evidence_links import rewrite_evidence_links
from tools.artifacts.doc_generation.model_cards import (
    render_model_card,
)
from tools.artifacts.doc_generation.support import (
    render_capability_evidence,
    render_support,
)


def expected_outputs(
    root: Path,
    registry: ModelRegistry,
    *,
    esmc_evidence: EsmcReportSet | None = None,
) -> dict[Path, str]:
    """Return every generated path and its deterministic UTF-8 content."""

    output = {
        root / "docs" / "generated" / "support.md": render_support(registry),
        root / "docs" / "generated" / "capability_evidence.md": render_capability_evidence(
            registry,
            esmc_evidence=esmc_evidence,
        ),
    }
    for spec in registry.values():
        output[root / "model_cards" / f"{spec.id}.md"] = render_model_card(
            spec,
            esmc_evidence=esmc_evidence,
            evidence_root=root,
        )
    return {
        path: rewrite_evidence_links(content, root=root, document_path=path)
        for path, content in output.items()
    }


def synchronize(
    root: Path,
    *,
    check: bool,
    esmc_report_root: Path | None = None,
    require_esmc_release_evidence: bool = False,
) -> list[str]:
    """Write generated files or return descriptions of stale files."""

    registry = get_model_registry()
    esmc_evidence = None
    if esmc_report_root is not None or require_esmc_release_evidence:
        selected_root = esmc_report_root
        if selected_root is None:
            selected_root = Path(
                os.environ.get(
                    "FASTPLMS_DIAGNOSTIC_REPORTS",
                    "artifacts/diagnostics/esmc",
                )
            )
        if not selected_root.is_absolute():
            selected_root = root / selected_root
        esmc_evidence = load_esmc_report_set(
            selected_root,
            registry,
            source_root=root,
        )
    outputs = expected_outputs(root, registry, esmc_evidence=esmc_evidence)
    failures: list[str] = []
    for path, content in outputs.items():
        rendered = content.rstrip() + "\n"
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current == rendered:
            continue
        if check:
            failures.append(f"stale or missing generated file: {path.relative_to(root)}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered, encoding="utf-8", newline="\n")

    expected_cards = {path.resolve() for path in outputs if path.parent.name == "model_cards"}
    for path in sorted((root / "model_cards").glob("*.md")):
        if path.name == "README.md" or path.resolve() in expected_cards:
            continue
        try:
            generated = GENERATED_MARKER in path.read_text(encoding="utf-8")
        except OSError:
            generated = False
        if generated and check:
            failures.append(f"stale generated model card: {path.relative_to(root)}")
        elif generated:
            path.unlink()
    return failures
