"""Fail-closed Hugging Face license metadata contracts for model cards."""

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

SEQUENCE_TTT_AUTO_CLASSES = {
    "ankh": "AutoModelForMaskedLM",
    "dplm": "AutoModelForMaskedLM",
    "dplm2": "AutoModelForMaskedLM",
    "e1": "AutoModelForMaskedLM",
    "esm2": "AutoModelForMaskedLM",
    "esm3": "AutoModel",
    "esm_plusplus": "AutoModelForMaskedLM",
}

EMBEDDING_FAMILIES = {
    "ankh",
    "dplm",
    "dplm2",
    "e1",
    "esm2",
    "esm3",
    "esm_plusplus",
    "esmfold2",
}


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


def test_every_manifest_model_card_has_task_oriented_guidance() -> None:
    registry = load_model_registry()
    specs = tuple(registry.values())
    assert len(specs) == 29

    for spec in specs:
        card = (ROOT / "model_cards" / f"{spec.id}.md").read_text(encoding="utf-8")
        normalized = " ".join(card.split())
        assert "## Model overview" in card
        assert "See Technical details for each registered class" in card
        assert "## Capabilities" not in card
        assert "| Feature | Status |" not in card
        assert "## Technical details" in card
        assert "## Validation and provenance" in card

        for backend in spec.family.attention:
            assert f"`{backend}`" in card
        assert "Requesting an unavailable backend raises" in normalized
        if "compliance" in spec.family.test_tiers:
            assert "Release validation includes the `compliance` tier." in card
        else:
            assert "does not declare the `compliance` tier" in card

        advertises_heads = {
            "AutoModelForSequenceClassification",
            "AutoModelForTokenClassification",
        }.issubset(spec.auto_map)
        assert "## PEFT fine-tuning" in card
        assert 'target_modules="all-linear"' in card
        assert "ESM2-specific shipped CLI is an example, not a\nsupport boundary" in card
        if advertises_heads:
            assert "## Downstream prediction" in card
            assert "their task heads are newly initialized" in card
            assert "create a new, untrained `classifier`" in card
            assert "AutoModelForSequenceClassification.from_pretrained" in card
            assert "AutoModelForTokenClassification.from_pretrained" in card
            assert "token_labels = torch.full_like(batch[\"input_ids\"], -100)" in card
            assert 'modules_to_save=["classifier"]' in card
        else:
            assert "## Downstream prediction" not in card
            assert "preserve any new head through `modules_to_save`" in card
            assert 'modules_to_save=["classifier"]' not in card

        if spec.family.id in EMBEDDING_FAMILIES:
            if spec.family.id == "esmfold2":
                assert "## Learned representation and ESMC precision" in card
            else:
                assert "## Dataset embeddings" in card
        else:
            assert "## Dataset embeddings" not in card
            assert "## Learned representation and ESMC precision" not in card

        ttt_auto_class = SEQUENCE_TTT_AUTO_CLASSES.get(spec.family.id)
        if ttt_auto_class is not None:
            section = card.split("## Test-time training", maxsplit=1)[1]
            assert f"from transformers import {ttt_auto_class}" in section
            assert "updates only injected low-rank" in section
            assert 'save_pretrained("adapted", safe_serialization=True)' in section
            assert "ttt_model.ttt_reset()" in section


def test_model_card_titles_use_readable_checkpoint_names() -> None:
    expected_titles = {
        "ankh_base": "# ANKH-Base",
        "ankh2_large": "# ANKH2-Large",
        "ankh3_xl": "# ANKH3-XL",
        "esm3_small": "# ESM3 Small",
        "esmc_small": "# ESM++ Small",
        "esmc_6b": "# ESM++ 6B",
    }
    for model_id, title in expected_titles.items():
        card = (ROOT / "model_cards" / f"{model_id}.md").read_text(encoding="utf-8")
        assert f"\n{title}\n" in card


def test_esmfold2_cards_publish_embedding_projection_shapes_and_ttt_scope() -> None:
    registry = load_model_registry()
    for spec in registry.by_family("esmfold2"):
        card = (ROOT / "model_cards" / f"{spec.id}.md").read_text(encoding="utf-8")
        assert "`H: (b, l, 81, 2560) -> Z: (b, l, 256)`" in card
        assert "returns one `(l, 256)` residue" in card
        if "experimental" in spec.id:
            assert "does not expose folding TTT" in card
        else:
            assert "## Optional folding TTT" in card
            assert "fold_protein_ttt(" in card
            assert "not a generic\n`save_pretrained` adapter-persistence path" in card


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
        assert f"Inputs: {spec.family.public_input}" in card
        assert "Input mode:" not in card
        assert "Internal preparation mode:" not in card
        assert f"Generation contract: `{spec.generation_contract}`" in card
        if spec.notes and spec.family.id != "esm_plusplus":
            assert "## Notes and limitations" in card
            assert " ".join(spec.notes.split()) in " ".join(card.split())
        elif spec.family.id == "esm_plusplus":
            # ESMC measurements are report-derived per checkpoint. The shared
            # manifest note contains a historical ESMC-6B-only observation
            # and must never be copied into the small or large checkpoint cards.
            assert "ESMC-6B Flex Attention exceeds" not in card


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
