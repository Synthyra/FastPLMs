"""CPU checks for optional confidence-head registry metadata and card text."""

from __future__ import annotations

import json
import pytest
from dataclasses import replace
from pathlib import Path

from fastplms.registry import (
    ConfidenceAdaptation,
    RegistryError,
    _parse_confidence_adaptation,
)
from tools.artifacts.doc_generation.model_cards import render_model_card


def _record() -> dict[str, str]:
    digest = "a" * 64
    return {
        "head_sha256": digest,
        "base_weight_sha256": "b" * 64,
        "donor_repo": "biohub/ESMFold2-Experimental-Fast-Cutoff2025",
        "donor_revision": "c" * 40,
        "donor_weight_sha256": "d" * 64,
        "training_url": "https://wandb.ai/lhallee/fastplms-confidence/runs/train",
        "evaluation_url": "https://wandb.ai/lhallee/fastplms-confidence/runs/evaluate",
        "evidence_path": "artifacts/confidence/evaluation.json",
    }


def test_confidence_adaptation_is_optional() -> None:
    assert _parse_confidence_adaptation({}, "models[0]") is None


def test_confidence_adaptation_rejects_nonportable_evidence_path() -> None:
    record = _record()
    record["evidence_path"] = "../evaluation.json"
    with pytest.raises(RegistryError, match="evidence_path"):
        _parse_confidence_adaptation({"confidence_adaptation": record}, "models[0]")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("donor_revision", "not-a-commit"),
        ("training_url", "https://wandb.ai/lhallee/fastplms-confidence/runs/train?x=1"),
    ),
)
def test_confidence_adaptation_rejects_unpinned_sources(field: str, value: str) -> None:
    record = _record()
    record[field] = value
    with pytest.raises(RegistryError, match=field if field != "training_url" else "W&B"):
        _parse_confidence_adaptation({"confidence_adaptation": record}, "models[0]")


def test_confidence_adaptation_rejects_non_esmfold2_model() -> None:
    with pytest.raises(RegistryError, match="only supported"):
        _parse_confidence_adaptation({"confidence_adaptation": _record()}, "models[0]", "esm2_8m")


@pytest.mark.parametrize("model_id", ("esmfold2_300", "esmfold2_600"))
def test_adapted_model_card_reads_checked_in_evidence(tmp_path: Path, model_id: str) -> None:
    from fastplms.registry import get_model_registry

    adaptation = ConfidenceAdaptation(**_record())
    evidence_path = tmp_path / adaptation.evidence_path
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text(
        json.dumps(
            {
                "candidate": {
                    "target_count": 128,
                    "atom_mae": 0.068,
                    "ca_mae": 0.069,
                    "calibration_error_10bin": 0.033,
                    "target_plddt_spearman": 0.91,
                    "iptm_dockq_spearman": 0.83,
                }
            }
        ),
        encoding="utf-8",
    )
    spec = replace(get_model_registry()[model_id], confidence_adaptation=adaptation)
    card = render_model_card(spec, evidence_root=tmp_path)
    assert "Synthyra-adapted native confidence head" in card
    assert "held-out evaluation used 128 targets" in card
    assert "64 monomers and 64 dimers" in card
    assert "calculate_confidence=False" in card
    assert "atom MAE 0.068" in card
    assert "Two-seed pLDDT sample selection: recorded" in card
    assert "two seeds only" in card
    assert "## Separately trained confidence head" not in card
    assert "confidence head disabled" not in card
    assert "The confidence head is disabled" not in card


def test_unadapted_cards_keep_existing_family_paths() -> None:
    from fastplms.registry import get_model_registry

    registry = get_model_registry()
    assert "The confidence fields are unavailable" not in render_model_card(registry["esm2_8m"])
    assert "The confidence head is disabled" not in render_model_card(registry["esmfold2"])


@pytest.mark.parametrize("model_id", ("esmfold2_300", "esmfold2_600"))
def test_unadapted_cards_withhold_metrics_pending_recomputation(model_id: str) -> None:
    from fastplms.registry import get_model_registry

    root = Path(__file__).resolve().parents[2]
    card = render_model_card(get_model_registry()[model_id], evidence_root=root)
    assert "## Separately trained confidence head" in card
    assert "This checkpoint ships with its confidence head disabled" in card
    assert "require recomputation" in " ".join(card.split())
    assert "780 update rows" in card
    assert "zero skipped training targets" in card
    assert "| Measurement | This head |" not in card
    assert "| Agreement with production" not in card
