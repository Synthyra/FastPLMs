"""Documentation evidence review and immutable hyperlink contracts."""

from __future__ import annotations

import hashlib
import json
import pytest

from dataclasses import replace
from pathlib import Path

from fastplms.registry import ConfidenceAdaptation, get_model_registry
from tools.artifacts.doc_generation.confidence_evidence import load_confidence_evidence
from tools.artifacts.doc_generation.confidence_rendering import (
    _confidence_adaptation_section,
    _confidence_research_section,
)
from tools.artifacts.doc_generation.evidence_links import (
    evidence_reference,
    rewrite_evidence_links,
)


def _evidence(root: Path, payload: object) -> Path:
    path = root / "docs/evidence/confidence/esmfold2_300-v2.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize("status", ("pending", "invalidated", "requires_recomputation"))
def test_all_confidence_sections_withhold_unreviewed_metrics(tmp_path: Path, status: str) -> None:
    path = _evidence(
        tmp_path,
        {"metrics_review": {"status": status}, "candidate": {"atom_mae": 0.987654}},
    )
    spec = replace(get_model_registry()["esmfold2_300"], confidence_adaptation=None)
    adaptation = ConfidenceAdaptation(
        head_sha256="a" * 64,
        base_weight_sha256="b" * 64,
        donor_repo="owner/donor",
        donor_revision="c" * 40,
        donor_weight_sha256="d" * 64,
        training_url="https://wandb.ai/owner/project/runs/training",
        evaluation_url="https://wandb.ai/owner/project/runs/evaluation",
        evidence_path=path.relative_to(tmp_path).as_posix(),
    )
    sections = (
        _confidence_research_section(spec, tmp_path),
        _confidence_adaptation_section(replace(spec, confidence_adaptation=adaptation), tmp_path),
    )
    for section in sections:
        assert f"`{status}`" in section
        assert "withheld" in section
        assert "0.988" not in section
        assert "| Measurement |" not in section
        assert "Held-out evidence:" not in section


def test_pilot_and_v2_have_distinct_unreviewed_evidence_policies(tmp_path: Path) -> None:
    path = _evidence(tmp_path, {"candidate": {"atom_mae": 0.1}})
    assert load_confidence_evidence(path, protocol="pilot").can_report_metrics
    assert not load_confidence_evidence(path, protocol="v2").can_report_metrics


@pytest.mark.parametrize("status", ("validated", "corrected_test_metrics"))
def test_reviewed_metrics_are_reportable(tmp_path: Path, status: str) -> None:
    path = _evidence(tmp_path, {"metrics_review": {"status": status}})
    assert load_confidence_evidence(path, protocol="v2").can_report_metrics


def test_card_rejects_tampered_declared_confidence_review(tmp_path: Path) -> None:
    path = _evidence(tmp_path, {"metrics_review": {"status": "pending"}})
    encoded = path.read_bytes() + b"          "
    path.write_bytes(encoded)
    relative = path.relative_to(tmp_path).as_posix()
    (tmp_path / "evidence.toml").write_text(
        'schema_version = 1\nrepository = "Synthyra/FastPLMs-artifacts"\n'
        f'revision = "{"b" * 40}"\n\n[[files]]\n'
        f'path = "{relative}"\nsize = {len(encoded)}\n'
        f'sha256 = "{hashlib.sha256(encoded).hexdigest()}"\n\n[[files]]\n'
        'path = "docs/evidence/unrelated.json"\nsize = 1\n'
        f'sha256 = "{"a" * 64}"\n',
        encoding="utf-8",
    )
    spec = replace(get_model_registry()["esmfold2_300"], confidence_adaptation=None)
    # A single card needs its own declared payload, not every file in the evidence store.
    assert "`pending`" in _confidence_research_section(spec, tmp_path)
    changed = encoded.replace(b"pending", b"validated").rstrip().ljust(len(encoded), b" ")
    path.write_bytes(changed)
    with pytest.raises(ValueError, match="Confidence evidence SHA-256 mismatch"):
        _confidence_research_section(spec, tmp_path)


@pytest.mark.parametrize(
    "payload",
    (
        [],
        {"metrics_review": None},
        {"metrics_review": []},
        {"metrics_review": {}},
        {"metrics_review": {"status": "unknown"}},
        {"metrics_review": {"status": "legacy_pilot"}},
    ),
)
def test_malformed_confidence_review_fails_closed(tmp_path: Path, payload: object) -> None:
    with pytest.raises(ValueError):
        load_confidence_evidence(_evidence(tmp_path, payload), protocol="v2")


def test_corrected_card_reports_undefined_metrics_and_spent_test_scope(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    payload = json.loads(
        (root / "docs/evidence/confidence/esmfold2_300-v2.json").read_text(encoding="utf-8")
    )
    payload["metrics_review"] = {"status": "corrected_test_metrics"}
    payload["test"]["heads"]["v2"]["plddt_lddt_spearman"] = None
    payload["test"]["heads"]["v2"]["interval_95"]["plddt_lddt_spearman"] = None
    payload["production_agreement"]["v2"]["mean_plddt_spearman"] = None
    _evidence(tmp_path, payload)
    spec = replace(get_model_registry()["esmfold2_300"], confidence_adaptation=None)
    section = _confidence_research_section(spec, tmp_path)
    assert "| pLDDT against all-atom lDDT, Spearman | undefined |" in section
    assert "| Mean pLDDT | undefined |" in section
    assert "The test split is spent" in section
    assert "not a new held-out evaluation" in " ".join(section.split())
    assert "validation correlation remains unverified" in " ".join(section.split())


def _manifest(root: Path, revision: str) -> None:
    (root / "evidence.toml").write_text(
        'schema_version = 1\nrepository = "Synthyra/FastPLMs-artifacts"\n'
        f'revision = "{revision}"\n\n[[files]]\n'
        'path = "docs/evidence/example.json"\nsize = 0\n'
        f'sha256 = "{"a" * 64}"\n',
        encoding="utf-8",
    )


def test_evidence_links_use_only_pinned_manifest_payloads(tmp_path: Path) -> None:
    revision = "b" * 40
    _manifest(tmp_path, revision)
    url = (
        f"https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/resolve/{revision}/"
        "docs/evidence/example.json"
    )
    markdown = (
        "[relative](../evidence/example.json)\n"
        "[root](/docs/evidence/example.json#record)\n"
        "[github](https://github.com/Synthyra/FastPLMs/blob/main/docs/evidence/example.json)\n"
        "[fixture](../../tests/fixtures/example.json)\n"
        "[external](https://example.org/docs/evidence/example.json)\n"
        "```python\ntext = '[code](../evidence/example.json)'\n```\n"
    )
    result = rewrite_evidence_links(
        markdown, root=tmp_path, document_path=tmp_path / "docs/generated/support.md"
    )
    assert f"[relative]({url})" in result
    assert f"[root]({url}#record)" in result
    assert f"[github]({url})" in result
    assert "[fixture](../../tests/fixtures/example.json)" in result
    assert "[external](https://example.org/docs/evidence/example.json)" in result
    assert "text = '[code](../evidence/example.json)'" in result
    assert evidence_reference("docs/evidence/example.json", tmp_path).endswith(f"]({url})")


def test_pending_evidence_keeps_existing_links(tmp_path: Path) -> None:
    _manifest(tmp_path, "pending")
    markdown = "[record](../evidence/example.json)"
    assert (
        rewrite_evidence_links(
            markdown, root=tmp_path, document_path=tmp_path / "docs/generated/support.md"
        )
        == markdown
    )
    assert (
        evidence_reference("docs/evidence/example.json", tmp_path) == "`docs/evidence/example.json`"
    )
