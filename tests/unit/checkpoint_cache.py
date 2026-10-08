"""The cached snapshot of a pinned checkpoint, found without a download, and the exact-load check run on it."""

from __future__ import annotations

import torch

from pathlib import Path
from safetensors.torch import load_file
from transformers import PreTrainedModel

from fastplms.registry import ModelSpec
from tools.artifacts.build import ArtifactError, verify_checkpoint


def find_verified_snapshot(spec: ModelSpec) -> Path | None:
    """The cached snapshot whose files match the manifest's pinned digests, or None when no cached revision does.

    The pinned revision is tried first. Another cached revision qualifies only when every pinned file in it hashes
    to the pinned digest, so it holds the same checkpoint bytes. The snapshot directories are listed directly:
    `huggingface_hub.scan_cache_dir` drops a whole repository when one of its refs names a commit with no local
    snapshot, which hides a complete pinned snapshot beside a stale `refs/main`.
    """

    from huggingface_hub import constants

    repository = f"models--{spec.fast.repo_id.replace('/', '--')}"
    snapshots_root = Path(constants.HF_HUB_CACHE) / repository / "snapshots"
    if not snapshots_root.is_dir():
        return None
    snapshots = sorted(
        snapshots_root.iterdir(), key=lambda snapshot: snapshot.name != spec.fast.revision
    )
    for snapshot in snapshots:
        try:
            verify_checkpoint(snapshot, spec.fast)
        except ArtifactError:
            continue
        return snapshot
    return None


def assert_checkpoint_loads_exactly(
    model_class: type[PreTrainedModel],
    snapshot: Path,
    weights_name: str = "model.safetensors",
) -> PreTrainedModel:
    """Load ``snapshot`` offline and require that every saved tensor reaches the model unchanged and nothing else does.

    For a model that saves every state name, with no tied weights left out of the file. The loader reports no missing,
    unexpected, or mismatched key. The model's state holds exactly the saved names, every floating tensor is finite,
    and each equals its saved tensor once read in the saved dtype, so a tensor left at its initial value or filled from
    uninitialized memory cannot pass.
    """

    model, loading_info = model_class.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.float32,
        output_loading_info=True,
    )
    for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        assert not loading_info[field], f"{snapshot.name}: the loader reported {field} {sorted(loading_info[field])[:5]}"
    saved = load_file(snapshot / weights_name)
    state = model.state_dict()
    assert set(state) == set(saved), (
        f"{snapshot.name}: model-only names {sorted(set(state) - set(saved))[:5]}, "
        f"file-only names {sorted(set(saved) - set(state))[:5]}"
    )
    for name, expected in saved.items():
        actual = state[name]
        if actual.is_floating_point():
            assert torch.isfinite(actual).all(), f"{snapshot.name}: {name} holds a non-finite value"
        assert torch.equal(actual.to(expected.dtype), expected), f"{snapshot.name}: {name} differs from the saved tensor"
    return model.eval()
