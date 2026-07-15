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
        assert f"Generation contract: `{spec.generation_contract}`" in card
        if spec.notes:
            assert "## Notes and limitations" in card
            assert spec.notes in card


def test_support_table_exposes_generation_contracts() -> None:
    registry = load_model_registry()
    support = render_support(registry)
    assert "| Generation contract |" in support
    dplm2_3b_row = next(line for line in support.splitlines() if line.startswith("| `dplm2_3b`"))
    assert "`official_unavailable`" in dplm2_3b_row
