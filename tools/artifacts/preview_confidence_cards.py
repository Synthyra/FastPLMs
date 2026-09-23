"""Generate local v1 model-card drafts without changing release metadata or the Hub."""

from __future__ import annotations

import argparse
import re

from dataclasses import replace
from pathlib import Path

from fastplms.registry import ConfidenceAdaptation, get_model_spec
from tools.artifacts.doc_generation.confidence_evidence import load_confidence_evidence
from tools.artifacts.doc_generation.model_cards import render_model_card


MODEL_IDS = ("esmfold2_300", "esmfold2_600")
DRAFT_NOTICE = (
    "> **Local review draft.** Examples describe the proposed release. "
    "Hub files have not been updated.\n\n"
)


def preview_card(model_id: str, evidence_path: Path, output_root: Path) -> Path:
    """Bind a draft to the evaluated head and its exact frozen training base."""
    if model_id not in MODEL_IDS:
        raise ValueError(f"Unsupported confidence model: {model_id}")
    report = load_confidence_evidence(evidence_path, protocol="v1")
    evidence = report.payload
    if not report.can_report_metrics or evidence.get("model_id") != model_id:
        raise ValueError("A draft requires reviewed evidence for the requested model")
    head = evidence["head_sha256"]
    if not isinstance(head, str) or re.fullmatch(r"[0-9a-f]{64}", head) is None:
        raise ValueError("A draft requires the exact evaluated head SHA-256")
    spec = get_model_spec(model_id)
    base = spec.confidence_training_base
    frozen = evidence["frozen_base"]
    expected_files = {
        item.path: (item.algorithm, item.digest) for item in base.files
    }
    recorded_files = {
        item["path"]: (item["algorithm"], item["digest"]) for item in frozen["files"]
    }
    if (
        frozen["repo_id"] != base.repo_id
        or frozen["revision"] != base.revision
        or recorded_files != expected_files
    ):
        raise ValueError("Evaluation evidence does not match the frozen training base")
    relative = f"evidence/{model_id}-v1.json"
    donor = evidence["donor"]
    adaptation = ConfidenceAdaptation(
        head_sha256=head,
        base_weight_sha256=base.file_map["model.safetensors"].digest,
        donor_repo=donor["repo_id"],
        donor_revision=donor["revision"],
        donor_weight_sha256=donor["weight_sha256"],
        training_url=evidence["training"]["wandb_url"],
        evaluation_url=f"[Local evaluation evidence](../{relative})",
        evidence_path=relative,
        release="v1",
        frozen_base=base,
    )
    draft = replace(
        spec,
        confidence_adaptation=adaptation,
        notes=(
            "Experimental Fast model with a frozen ESM++ backbone, 24 folding blocks, "
            "and no MSA conditioning. BF16 execution uses FP32 folding parameters "
            "with CUDA autocast; FP8 is unsupported. The confidence evaluation above "
            "does not establish full structure-model equivalence to production ESMFold2."
        ),
    )
    stored_evidence = output_root / relative
    stored_evidence.parent.mkdir(parents=True, exist_ok=True)
    stored_evidence.write_bytes(evidence_path.read_bytes())
    card = render_model_card(draft, evidence_root=output_root)
    title = f"# {spec.fast.repo_id.rsplit('/', maxsplit=1)[-1]}\n\n"
    card = card.replace(title, title + DRAFT_NOTICE, 1)
    destination = output_root / "model_cards" / f"{model_id}.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(card.rstrip() + "\n", encoding="utf-8", newline="\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    for model_id in MODEL_IDS:
        path = arguments.evidence_root / f"{model_id}-v1-preview.json"
        print(preview_card(model_id, path, arguments.output_root))


if __name__ == "__main__":
    main()
