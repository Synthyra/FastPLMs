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
    get_model_registry,
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


def _v1_record() -> dict[str, object]:
    return {
        **_record(),
        "release": "v1",
        "frozen_base": {
            "repo": "Synthyra/ESMFold2-300",
            "revision": "e" * 40,
            "files": [
                "config.json=git-sha1:" + "f" * 40,
                "model.safetensors=sha256:" + "b" * 64,
            ],
        },
    }


def test_pilot_adaptation_keeps_legacy_defaults() -> None:
    adaptation = _parse_confidence_adaptation({"confidence_adaptation": _record()}, "models[0]")
    assert adaptation is not None
    assert adaptation.release == "pilot"
    assert adaptation.frozen_base is None
    spec = replace(get_model_registry()["esmfold2_300"], confidence_adaptation=None)
    assert spec.confidence_training_base == spec.fast
    assert replace(spec, confidence_adaptation=adaptation).confidence_training_base == spec.fast


def test_v1_preserves_training_base_after_published_revision_changes() -> None:
    adaptation = _parse_confidence_adaptation({"confidence_adaptation": _v1_record()}, "models[0]")
    assert adaptation is not None
    assert adaptation.release == "v1"
    assert adaptation.frozen_base is not None
    assert adaptation.frozen_base.repo_id == "Synthyra/ESMFold2-300"
    assert adaptation.frozen_base.revision == "e" * 40
    assert adaptation.frozen_base.file_map["config.json"].digest == "f" * 40
    assert (
        adaptation.frozen_base.file_map["model.safetensors"].digest == adaptation.base_weight_sha256
    )
    spec = get_model_registry()["esmfold2_300"]
    published = replace(spec.fast, revision="1" * 40)
    released = replace(spec, fast=published, confidence_adaptation=adaptation)
    assert released.artifact_checkpoint == published
    assert released.confidence_training_base == adaptation.frozen_base
    assert released.confidence_training_base != released.fast


@pytest.mark.parametrize("release", ("v2", "", None, [], 1))
def test_confidence_adaptation_rejects_unknown_release(release: object) -> None:
    record = {**_record(), "release": release}
    with pytest.raises(RegistryError, match="release"):
        _parse_confidence_adaptation({"confidence_adaptation": record}, "models[0]")


def test_v1_requires_a_frozen_base() -> None:
    record = {**_record(), "release": "v1"}
    with pytest.raises(RegistryError, match="frozen_base is required"):
        _parse_confidence_adaptation({"confidence_adaptation": record}, "models[0]")


def test_v1_requires_the_exact_training_config() -> None:
    record = _v1_record()
    record["frozen_base"]["files"] = ["model.safetensors=sha256:" + "b" * 64]
    with pytest.raises(RegistryError, match="config.json"):
        _parse_confidence_adaptation({"confidence_adaptation": record}, "models[0]")


@pytest.mark.parametrize(
    "base",
    (
        None,
        {
            "repo": "Synthyra/ESMFold2-300",
            "revision": "main",
            "files": ["model.safetensors=sha256:" + "b" * 64],
        },
        {
            "repo": "Synthyra/ESMFold2-300",
            "revision": "e" * 40,
            "files": ["model.safetensors=sha256:" + "c" * 64],
        },
        {
            "repo": "Synthyra/ESMFold2-300",
            "revision": "e" * 40,
            "files": ["model.safetensors=git-sha1:" + "b" * 40],
        },
        {
            "repo": "Synthyra/ESMFold2-300",
            "revision": "e" * 40,
            "files": ["pytorch_model.bin=sha256:" + "b" * 64],
        },
    ),
)
def test_v1_rejects_invalid_frozen_base(base: object) -> None:
    record = {**_v1_record(), "frozen_base": base}
    with pytest.raises(RegistryError, match="frozen_base"):
        _parse_confidence_adaptation({"confidence_adaptation": record}, "models[0]")


@pytest.mark.parametrize(
    "view_path",
    (
        "tree/{revision}",
        "tree/{revision}/confidence/v1",
        "blob/{revision}/confidence/v1/report.json",
    ),
)
def test_evaluation_accepts_pinned_synthyra_dataset_url(view_path: str) -> None:
    url = "https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/" + view_path.format(
        revision="a" * 40
    )
    record = {**_v1_record(), "evaluation_url": url}
    adaptation = _parse_confidence_adaptation({"confidence_adaptation": record}, "models[0]")
    assert adaptation is not None
    assert adaptation.evaluation_url == url


@pytest.mark.parametrize(
    "url",
    (
        "https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/tree/main/confidence",
        "https://huggingface.co/datasets/other/artifacts/tree/" + "a" * 40,
        "https://huggingface.co/Synthyra/ESMFold2-300/tree/" + "a" * 40,
        "https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/blob/" + "a" * 40,
        "https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/tree/"
        + "a" * 40
        + "?download=true",
        "https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/tree/" + "a" * 40 + "#metrics",
        "https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/tree/" + "a" * 40 + "/../main",
    ),
)
def test_evaluation_rejects_unpinned_or_unscoped_dataset_url(url: str) -> None:
    record = {**_v1_record(), "evaluation_url": url}
    with pytest.raises(RegistryError, match="evaluation_url"):
        _parse_confidence_adaptation({"confidence_adaptation": record}, "models[0]")


def test_training_url_still_requires_wandb() -> None:
    record = {
        **_v1_record(),
        "training_url": "https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/tree/"
        + "a" * 40,
    }
    with pytest.raises(RegistryError, match="training_url.*W&B"):
        _parse_confidence_adaptation({"confidence_adaptation": record}, "models[0]")


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
    spec = replace(get_model_registry()[model_id], confidence_adaptation=None)
    card = render_model_card(spec, evidence_root=root)
    assert "## Separately trained confidence head" in card
    assert "The pinned base checkpoint has its confidence head disabled" in card
    assert "Current v2 confidence head" in card
    assert "pending evaluation" in card
    assert "require recomputation" in " ".join(card.split())
    assert "780 update rows" in card
    assert "zero skipped training targets" in card
    assert "| Measurement | This head |" not in card
    assert "| Agreement with production" not in card
