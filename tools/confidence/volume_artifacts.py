"""Select archived confidence inputs before research workers allocate GPUs."""

from __future__ import annotations

import hashlib
import json
import tempfile

from pathlib import Path, PurePosixPath


ARTIFACT_VOLUME = "synthyra-experiment-artifacts"
HF_VOLUME = "synthyra-hf-cache"
ARTIFACT_MOUNT = Path("/artifacts")
HF_MOUNT = Path("/hf")
PILOT_SOURCE = "fastplms-confidence-pilot"
V2_SOURCE = "fastplms-confidence-v2"
PILOT_ROOT = ARTIFACT_MOUNT / PILOT_SOURCE / "confidence"
V2_ROOT = ARTIFACT_MOUNT / V2_SOURCE


def restore_source(source: str, prefixes: tuple[str, ...]) -> None:
    """Restore only named immutable prefixes; discard downloaded tar shards afterwards."""
    from foundry.artifact_archive import restore_files

    if source not in {PILOT_SOURCE, V2_SOURCE}:
        raise ValueError(f"Unknown confidence archive source: {source}")
    if not prefixes or any(not prefix or ".." in PurePosixPath(prefix).parts or PurePosixPath(prefix).is_absolute() or "\\" in prefix for prefix in prefixes):
        raise ValueError("Confidence restoration requires explicit safe prefixes")
    receipt = json.loads((HF_MOUNT / "volume_archive_receipts" / source / "complete.json").read_text())
    if receipt.get("status") != "verified" or receipt.get("source_volume") != source:
        raise ValueError(f"Missing verified archive receipt for {source}")
    missing = []
    for prefix in prefixes:
        identity = hashlib.sha256(f"{receipt['manifest_sha256']}:{prefix}".encode()).hexdigest()
        marker = ARTIFACT_MOUNT / ".archive_restored" / source / identity
        if not marker.is_file():
            missing.append((prefix, marker))
    if not missing:
        return
    with tempfile.TemporaryDirectory(prefix="confidence-archive-") as ephemeral:
        restore_files(receipt["repo_id"], receipt["manifest_path"], receipt["revision"], receipt["manifest_sha256"],
                      ARTIFACT_MOUNT / source, prefixes=tuple(prefix for prefix, _ in missing), cache_dir=ephemeral)
    for prefix, marker in missing:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"prefix": prefix, "manifest_sha256": receipt["manifest_sha256"]}) + "\n")


def pilot_prefixes(stage: str, model_id: str, options: dict[str, object]) -> tuple[str, ...]:
    prefixes = {"confidence/data/", "confidence/inspection/"}
    prefixes.update(f"confidence/archives/{source}.json" for source in ("cameo_val", "rcsb", "rcsb_multimer", "rcsb_multimer_val"))
    if stage in {"train", "evaluate", "campaign"}:
        prefixes.update({f"confidence/{model_id}/cache/", f"confidence/{model_id}/train/"})
    elif stage in {"package", "release"}:
        prefixes.update({f"confidence/{model_id}/train/", f"confidence/{model_id}/evaluate-report.json"})
    elif stage == "prepare":
        source = str(options.get("source", "cameo_val"))
        if source not in {"cameo_val", "rcsb", "rcsb_multimer", "rcsb_multimer_val"}:
            raise ValueError(f"Unknown prepared structure source: {source}")
        phase = str(options.get("phase", "download"))
        prefixes.add("confidence/atlasfold/" if phase in {"normalize", "smoke", "quality", "inspect"}
                     else f"confidence/atlasfold/{source}/")
    if options.get("smoke"):
        prefixes.add("confidence/smoke/")
    return tuple(sorted(prefixes))


def restore_v2(campaign: str, stage: str, model_id: str | None = None) -> None:
    from .experiment_artifacts import validate_evaluation_id

    campaign = validate_evaluation_id(campaign)
    if stage not in {"prepare", "start", "evaluate", "resume", "benchmark"}:
        return
    relative = campaign + "/"
    basic = ("prepared.json", "dataset.json", "splits/", "pilot/", "ledgers/")
    prefixes = [relative + path for path in basic]
    if model_id is not None:
        if model_id not in {"esmfold2_300", "esmfold2_600"}:
            raise ValueError("Select the 300M or 600M confidence head")
        prefixes.extend([f"{relative}runs/{model_id}/v2/", f"{relative}status/", f"{relative}evaluation/{campaign}/{model_id}/",
                         f"{relative}public/evaluation/esmfold2/"])
    if stage == "prepare":
        restore_source(PILOT_SOURCE, ("confidence/data/records.json", "confidence/esmfold2_300/train/best.safetensors",
                                      "confidence/esmfold2_600/train/best.safetensors"))
    restore_source(V2_SOURCE, tuple(prefixes))
    root = V2_ROOT / campaign
    if not (root / "prepared.json").exists():
        if stage == "prepare":
            return
        raise FileNotFoundError("No archived prepared campaign; run its prepare stage first")
    from .target_splits import load_split

    targets = load_split(root / "splits")
    if stage == "benchmark":
        from .online_training import TargetSampler
        from .gpu_benchmark import MEASURED_TARGETS, SAMPLER_SEED

        train = [target for target in targets if target["split"] == "train"]
        sampler = TargetSampler(train, 0.5, SAMPLER_SEED)
        chosen = [sampler.draw() for _ in range(MEASURED_TARGETS)]
        chosen.extend([min(train, key=lambda t: abs(int(t["num_tokens"]) - 256)),
                       min((t for t in train if int(t["num_chains"]) > 1), key=lambda t: abs(int(t["num_tokens"]) - 1024)),
                       min((t for t in targets if t["split"] == "unused" and t["variant"] == "long" and int(t["num_chains"]) > 1),
                           key=lambda t: abs(int(t["num_tokens"]) - 2048))])
    else:
        wanted = {"test"} if stage in {"evaluate", "resume"} else {"train", "validation", "test", "unused"}
        chosen = [target for target in targets if target["split"] in wanted]
    coordinates = {f"{relative}pool/positions/{target['positions_file']}" for target in chosen}
    restore_source(V2_SOURCE, tuple(sorted(coordinates)))
