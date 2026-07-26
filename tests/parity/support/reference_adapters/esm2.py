"""Load ESM2 through the pinned Meta ESM implementation."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
import torch
import torch.nn as nn
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tests.parity.support.reference_adapters import move_model, snapshot_path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_FAIR_ESM_SUBMODULE = _REPOSITORY_ROOT / "vendor" / "upstream" / "fair-esm"


def _asset_field(asset: object, name: str) -> Any:
    if isinstance(asset, Mapping):
        return asset[name]
    return getattr(asset, name)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_asset(asset: object) -> Path:
    relative = Path(str(_asset_field(asset, "path")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe fair-esm oracle asset path: {relative}")
    torch_home = Path(os.environ.get("TORCH_HOME", "~/.cache/torch")).expanduser()
    cache_root = Path(os.environ.get("FASTPLMS_ORACLE_CACHE", str(torch_home / "fair-esm")))
    destination = cache_root / relative
    expected_size = int(_asset_field(asset, "size"))
    expected_sha256 = str(_asset_field(asset, "sha256"))

    def valid() -> bool:
        return (
            destination.is_file()
            and destination.stat().st_size == expected_size
            and _file_sha256(destination) == expected_sha256
        )

    if valid():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        urllib.request.urlretrieve(str(_asset_field(asset, "url")), temporary_path)
        if temporary_path.stat().st_size != expected_size:
            raise RuntimeError(f"Size mismatch for fair-esm oracle asset {relative}")
        if _file_sha256(temporary_path) != expected_sha256:
            raise RuntimeError(f"SHA-256 mismatch for fair-esm oracle asset {relative}")
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


class _AlphabetTokenizer:
    """Expose the official Alphabet through the subset used by parity tests."""

    def __init__(self, alphabet: Any) -> None:
        self.alphabet = alphabet
        self.pad_token_id = alphabet.padding_idx
        self.cls_token_id = alphabet.cls_idx
        self.bos_token_id = alphabet.cls_idx
        self.eos_token_id = alphabet.eos_idx
        self.mask_token_id = alphabet.mask_idx
        self.unk_token_id = alphabet.unk_idx
        self.all_special_ids = [
            self.pad_token_id,
            self.cls_token_id,
            self.eos_token_id,
            self.mask_token_id,
            self.unk_token_id,
        ]

    def get_vocab(self) -> dict[str, int]:
        return dict(self.alphabet.tok_to_idx)

    def __call__(
        self,
        sequences: str | Sequence[str],
        *,
        return_tensors: str = "pt",
        padding: bool | str = True,
        truncation: bool = False,
        max_length: int | None = None,
        **_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError("The official ESM2 parity tokenizer returns PyTorch tensors")
        values = [sequences] if isinstance(sequences, str) else list(sequences)
        if truncation and max_length is not None:
            residue_limit = max(1, max_length - 2)
            values = [sequence[:residue_limit] for sequence in values]
        converter = self.alphabet.get_batch_converter()
        _, _, input_ids = converter(
            [(str(index), sequence) for index, sequence in enumerate(values)]
        )
        if padding == "max_length" and max_length is not None and input_ids.shape[1] < max_length:
            # pad: (input_ids.shape[0], max_length - input_ids.shape[1])
            pad = torch.full(
                (input_ids.shape[0], max_length - input_ids.shape[1]),
                self.pad_token_id,
                dtype=input_ids.dtype,
            )
            # input_ids: (...)
            input_ids = torch.cat((input_ids, pad), dim=1)
        # attention_mask: (b, l)
        attention_mask = input_ids.ne(self.pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _OfficialESM2ForwardWrapper(nn.Module):
    def __init__(self, model: nn.Module, alphabet: Any) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = _AlphabetTokenizer(alphabet)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_kwargs: Any,
    ) -> Any:
        # input_ids: (b, l)
        del attention_mask
        layers = list(range(self.model.num_layers + 1))
        output = self.model(input_ids, repr_layers=layers, return_contacts=False)
        hidden_states = tuple(output["representations"][index] for index in layers)
        return SimpleNamespace(
            logits=output["logits"],
            last_hidden_state=hidden_states[-1],
            hidden_states=hidden_states,
        )


def load_official_model(
    reference_repo_id: str,
    reference_revision: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
    oracle_assets: Sequence[object] = (),
) -> tuple[nn.Module, _AlphabetTokenizer]:
    """Load Meta's exact fair-esm code and hash-pinned native checkpoint."""

    if not _FAIR_ESM_SUBMODULE.is_dir():
        raise FileNotFoundError(
            "Meta ESM submodule is missing; run git submodule update --init --recursive"
        )
    source = str(_FAIR_ESM_SUBMODULE)
    if source not in sys.path:
        sys.path.insert(0, source)
    import esm

    # Resolve the declared immutable Hub snapshot as a provenance gate even
    # though fair-esm's native oracle consumes Meta's hash-pinned `.pt` files.
    snapshot_path(reference_repo_id, reference_revision)
    by_role = {str(_asset_field(asset, "role")): asset for asset in oracle_assets}
    if set(by_role) != {"weights", "contact_regression"}:
        raise RuntimeError(
            "ESM2 live parity requires hash-pinned weights and contact-regression assets"
        )
    weights_path = _verified_asset(by_role["weights"])
    regression_path = _verified_asset(by_role["contact_regression"])
    model_name = reference_repo_id.rsplit("/", 1)[-1]
    model_data = torch.load(weights_path, map_location="cpu", weights_only=False)
    regression_data = torch.load(regression_path, map_location="cpu", weights_only=False)
    model, alphabet = esm.pretrained.load_model_and_alphabet_core(
        model_name,
        model_data,
        regression_data,
    )
    wrapped = _OfficialESM2ForwardWrapper(model, alphabet)
    wrapped = move_model(wrapped, device, dtype).eval()
    return wrapped, wrapped.tokenizer
