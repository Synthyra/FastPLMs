"""Hidden-state sparse autoencoders for ESM++ (ESMC) models.

FastPLMs implements the published Biohub hidden-state SAE contract directly, so attaching an SAE
needs only PyTorch, Transformers, and the checkpoint itself. Biohub still owns the SAE weights:
this module reads their published repository layout, one shared ``config.json`` plus one
``layer_{index}.safetensors`` shard per backbone layer, and never redistributes those tensors.

A layer produced here satisfies the same attachment contract as an official Biohub
``ESMCSAEModel.layers`` entry, so ``PreTrainedESMplusplusModel.add_sae_models`` accepts either.
"""

from __future__ import annotations

import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import Tensor


SAE_CONFIG_FILE = "config.json"
SAE_REQUIRED_CONFIG_FIELDS = ("d_model", "codebook_dim", "k")
# Trained per-feature statistics. Shards that never trained them omit them, and the ones default
# makes the (features / max) * idf normalization an identity.
SAE_OPTIONAL_STATE_NAMES = ("idf", "max")
STANDARDIZATION_EPSILON = 1e-5


@dataclass(frozen=True, slots=True)
class ESMplusplusSAEParams:
    """Shape contract of one hidden-state SAE.

    ``layer`` indexes the ESM++ hidden state the SAE reads, where ``0`` is the embedding output and
    ``num_hidden_layers`` is the final normalized state.
    """

    d_model: int
    codebook_dim: int
    k: int
    layer: int


@dataclass(slots=True)
class ESMplusplusSAEOutput:
    """Sparse features, and the optional reconstruction error, for one batch of residues."""

    feature_magnitudes: Tensor
    reconstruction_loss: Tensor | None = None


def _standardize_residue_states(residue_states: Tensor) -> Tensor:
    """Center and scale each residue vector the way the SAEs were trained."""

    # residue_states: (n, d)
    centered = residue_states - residue_states.mean(dim=-1, keepdim=True)  # (n, d)
    return centered / (centered.std(dim=-1, keepdim=True) + STANDARDIZATION_EPSILON)  # (n, d)


class ESMplusplusSAELayer(nn.Module):
    """Top-k sparse autoencoder over one ESM++ hidden state.

    Encoding standardizes each residue vector, projects it into a wide codebook, and keeps only the
    ``k`` largest activations. The decoder is used only to measure reconstruction error, which is
    why it is opt-in: interpretation and gradient-based design read ``feature_magnitudes`` alone.
    """

    idf: Tensor
    max: Tensor

    def __init__(self, params: ESMplusplusSAEParams) -> None:
        super().__init__()
        self.params = params
        self.W_enc = nn.Parameter(torch.empty(params.d_model, params.codebook_dim))  # (d, c)
        self.W_dec = nn.Parameter(torch.empty(params.codebook_dim, params.d_model))  # (c, d)
        self.b_dec = nn.Parameter(torch.zeros(params.d_model))  # (d,)
        self.register_buffer("idf", torch.ones(params.codebook_dim))  # (c,)
        self.register_buffer("max", torch.ones(params.codebook_dim))  # (c,)

    @property
    def layer(self) -> int:
        """ESM++ hidden-state index this SAE was trained against."""

        return self.params.layer

    def forward(
        self,
        residue_states: Tensor,
        *,
        compute_reconstruction_loss: bool = False,
    ) -> ESMplusplusSAEOutput:
        # residue_states: (n, d) for n residues; c is the codebook width, k the retained features
        standardized = _standardize_residue_states(residue_states)  # (n, d)
        preactivations = F.relu((standardized - self.b_dec) @ self.W_enc)  # (n, c)
        retained = torch.topk(preactivations, self.params.k, dim=-1)  # values, indices: (n, k)
        feature_magnitudes = torch.zeros_like(preactivations).scatter(
            -1, retained.indices, retained.values
        )  # (n, c)
        if not compute_reconstruction_loss:
            return ESMplusplusSAEOutput(feature_magnitudes=feature_magnitudes)

        reconstructed = feature_magnitudes @ self.W_dec + self.b_dec  # (n, d)
        return ESMplusplusSAEOutput(
            feature_magnitudes=feature_magnitudes,
            reconstruction_loss=(reconstructed - standardized).pow(2).mean(dim=-1),  # (n,)
        )

    def get_sae_output(self, layer_states: Tensor, token_mask: Tensor) -> ESMplusplusSAEOutput:
        """Encode the unpadded residues of one ESM++ hidden state.

        This name and signature are the attachment contract that
        ``PreTrainedESMplusplusModel.add_sae_models`` validates and calls.
        """

        # layer_states: (b, l, d); token_mask: (b, l)
        residue_states = layer_states[token_mask]  # (n, d) for n valid tokens
        encoded: ESMplusplusSAEOutput = self(residue_states)
        return encoded


def _repository_file(
    repository: str | os.PathLike[str],
    filename: str,
    *,
    revision: str | None,
    cache_dir: str | os.PathLike[str] | None,
    token: str | bool | None,
    local_files_only: bool,
) -> Path:
    """Resolve one repository file, from a local directory or the Hub cache."""

    # A directory counts as local only when it holds the shared config, so a stale directory named
    # like a Hub identifier cannot shadow the download.
    local_directory = Path(repository)
    if (local_directory / SAE_CONFIG_FILE).is_file():
        path = local_directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"SAE repository {local_directory} has no {filename}.")
        return path

    return Path(
        hf_hub_download(
            repo_id=str(repository),
            filename=filename,
            revision=revision,
            cache_dir=None if cache_dir is None else str(cache_dir),
            token=token,
            local_files_only=local_files_only,
        )
    )


def _load_sae_layer(
    shard_path: Path,
    *,
    params: ESMplusplusSAEParams,
    device: torch.device,
    dtype: torch.dtype | None,
) -> ESMplusplusSAELayer:
    state = load_file(str(shard_path), device=str(device))
    encoder = state.get("W_enc")
    if encoder is None:
        raise ValueError(f"{shard_path} is not an ESMC SAE shard; it has no 'W_enc' entry.")

    # Build on the meta device so the shard tensors are the only materialized copy of a codebook
    # that reaches roughly one gigabyte for the widest published SAEs.
    with torch.device("meta"):
        sae_layer = ESMplusplusSAELayer(params)
    sae_layer.to(dtype=encoder.dtype if dtype is None else dtype)
    sae_layer.to_empty(device=device)
    # to_empty leaves the statistics buffers uninitialized, so restore the identity defaults that
    # shards without trained statistics rely on.
    sae_layer.idf.fill_(1.0)
    sae_layer.max.fill_(1.0)

    incompatible = sae_layer.load_state_dict(state, strict=False)
    missing = tuple(
        name for name in incompatible.missing_keys if name not in SAE_OPTIONAL_STATE_NAMES
    )
    unexpected = tuple(incompatible.unexpected_keys)
    if missing or unexpected:
        raise ValueError(
            f"{shard_path} does not match the ESMC SAE state contract; "
            f"missing {list(missing)}, unexpected {list(unexpected)}."
        )
    return sae_layer


def load_esmc_sae_layers(
    repository: str | os.PathLike[str],
    layers: Sequence[int],
    *,
    revision: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> dict[int, ESMplusplusSAELayer]:
    """Load the requested hidden-state SAE layers from a Hub repository or a local directory.

    Only the shared config and the requested shards are read, so a repository that publishes every
    backbone layer costs one shard per requested layer. ``dtype`` defaults to the dtype stored in
    the shard; pass the ESM++ model dtype when the SAE has to consume its hidden states directly.
    """

    requested = tuple(dict.fromkeys(int(layer) for layer in layers))
    if not requested:
        raise ValueError("Loading SAE layers requires at least one backbone layer index.")

    config_path = _repository_file(
        repository,
        SAE_CONFIG_FILE,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    missing_fields = [name for name in SAE_REQUIRED_CONFIG_FIELDS if name not in config]
    if missing_fields:
        raise ValueError(f"{config_path} is not an ESMC SAE config; it omits {missing_fields}.")
    available = tuple(int(index) for index in config.get("available_layers", ()))

    target_device = torch.device(device)
    sae_layers: dict[int, ESMplusplusSAELayer] = {}
    for layer in requested:
        if available and layer not in available:
            raise ValueError(
                f"SAE repository {repository} does not publish layer {layer}; "
                f"available layers are {list(available)}."
            )
        shard_path = _repository_file(
            repository,
            f"layer_{layer}.safetensors",
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        sae_layers[layer] = _load_sae_layer(
            shard_path,
            params=ESMplusplusSAEParams(
                d_model=int(config["d_model"]),
                codebook_dim=int(config["codebook_dim"]),
                k=int(config["k"]),
                layer=layer,
            ),
            device=target_device,
            dtype=dtype,
        )
    return sae_layers


__all__ = [
    "ESMplusplusSAELayer",
    "ESMplusplusSAEOutput",
    "ESMplusplusSAEParams",
    "load_esmc_sae_layers",
]
