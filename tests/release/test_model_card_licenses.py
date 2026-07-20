"""Fail-closed Hugging Face license metadata contracts for model cards."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from fastplms.registry import HUB_LICENSE_IDENTIFIERS, ModelSpec, load_model_registry
from tools.artifacts.build import render_model_card as render_artifact_model_card
from tools.artifacts.generate_docs import (
    render_model_card as render_documentation_model_card,
)
from tools.artifacts.generate_docs import (
    render_support,
)
from tools.artifacts.license_metadata import parse_hub_license_metadata

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "renderer",
    (render_documentation_model_card, render_artifact_model_card),
    ids=("documentation", "artifact-fallback"),
)
def test_model_card_renderers_use_typed_hub_license_metadata(
    renderer: Callable[[ModelSpec], str],
) -> None:
    registry = load_model_registry()
    for spec in registry.values():
        metadata = parse_hub_license_metadata(renderer(spec))
        assert metadata == dict(spec.family.hub_license_metadata)
        assert metadata["license"] in HUB_LICENSE_IDENTIFIERS


def test_checked_in_model_cards_use_typed_hub_license_metadata() -> None:
    registry = load_model_registry()
    for spec in registry.values():
        card = (ROOT / "model_cards" / f"{spec.id}.md").read_text(encoding="utf-8")
        assert parse_hub_license_metadata(card) == dict(spec.family.hub_license_metadata)


def test_model_cards_include_family_appropriate_usage() -> None:
    registry = load_model_registry()
    expected_sections = {
        "ankh": (
            "## Tokenization and forward inference",
            "## Encoder and sequence-to-sequence use",
        ),
        "boltz2": ("## Protein structure prediction",),
        "dplm": ("## Tokenization and forward inference", "## Diffusion sequence generation"),
        "dplm2": ("## Amino-acid and structure co-generation",),
        "e1": ("## Tokenizer-free E1 input",),
        "esm2": (
            "## Tokenization and forward inference",
            "## Masked language modeling and contacts",
        ),
        "esm3": ("## Sequence inference and masked-sequence generation",),
        "esm_plusplus": ("## Tokenization and forward inference", "## ESMC behavior"),
        "esmfold": ("## Protein structure prediction",),
        "esmfold2": ("## Protein folding", "## Learned representation and ESMC precision"),
    }
    stale_fragments = (
        "attn_backend",
        "kernels_flash",
        "pooling_types",
        "save_path=",
        'format="pth"',
    )
    for spec in registry.values():
        card = (ROOT / "model_cards" / f"{spec.id}.md").read_text(encoding="utf-8")
        assert "## Quick start" in card
        assert spec.fast.repo_id in card
        for section in expected_sections[spec.family.id]:
            assert section in card
        for fragment in stale_fragments:
            assert fragment not in card

        if spec.family.id in {
            "ankh",
            "dplm",
            "dplm2",
            "e1",
            "esm2",
            "esm3",
            "esm_plusplus",
        }:
            assert "## Dataset embeddings" in card


def test_model_cards_keep_checkpoint_specific_ttt_boundaries() -> None:
    esmfold = (ROOT / "model_cards" / "esmfold.md").read_text(encoding="utf-8")
    assert "does not expose ProteinTTT" in esmfold

    registry = load_model_registry()
    for spec in registry.by_family("esmfold2"):
        card = (ROOT / "model_cards" / f"{spec.id}.md").read_text(encoding="utf-8")
        if "experimental" in spec.id:
            assert "standard and Fast checkpoints expose" not in card
        else:
            assert "standard and Fast checkpoints expose" in card


@pytest.mark.parametrize(
    "renderer",
    (render_documentation_model_card, render_artifact_model_card),
    ids=("documentation", "artifact-fallback"),
)
def test_model_card_renderers_surface_typed_limitations(
    renderer: Callable[[ModelSpec], str],
) -> None:
    registry = load_model_registry()
    for spec in registry.values():
        card = renderer(spec)
        assert f"Public input: {spec.family.public_input}" in card
        assert "Input mode:" not in card
        assert "Internal preparation mode:" not in card
        assert f"Generation contract: `{spec.generation_contract}`" in card
        if spec.notes:
            assert "## Notes and limitations" in card
            assert " ".join(spec.notes.split()) in " ".join(card.split())


def test_support_table_exposes_generation_contracts() -> None:
    registry = load_model_registry()
    support = render_support(registry)
    assert "| Public input |" in support
    assert "| Input |" not in support
    for family in registry.families.values():
        assert family.public_input in support
    assert "| Generation contract |" in support
    dplm2_3b_row = next(line for line in support.splitlines() if line.startswith("| `dplm2_3b`"))
    assert "`official_unavailable`" in dplm2_3b_row
