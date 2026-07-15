"""Tokenizer loading and raw-sequence batch preparation for E1."""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass

import torch
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
E1_VOCAB_SIZE = 34
E1_TOKENIZER_REPO_ID = "Synthyra/Profluent-E1-150M"


def _load_tokenizer_file(fname: str) -> Tokenizer:
    tokenizer: Tokenizer = Tokenizer.from_file(fname)
    assert tokenizer.padding["pad_id"] == PAD_TOKEN_ID, (
        f"Padding token id must be {PAD_TOKEN_ID}, but got {tokenizer.padding['pad_id']}"
    )
    return tokenizer


def get_tokenizer(
    pretrained_model_name_or_path: str | os.PathLike | None = None,
    *,
    local_files_only: bool = False,
    cache_dir: str | os.PathLike | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
) -> Tokenizer:
    source_path = None
    checked_local_source = False
    if pretrained_model_name_or_path is not None:
        source_path = os.fspath(pretrained_model_name_or_path)
        if os.path.isdir(source_path):
            checked_local_source = True
            fname = os.path.join(source_path, "tokenizer.json")
            if os.path.isfile(fname):
                return _load_tokenizer_file(fname)

    fname = os.path.join(os.path.dirname(__file__), "tokenizer.json")
    if os.path.isfile(fname):
        return _load_tokenizer_file(fname)

    if local_files_only and checked_local_source:
        raise FileNotFoundError(
            f"E1 tokenizer.json was not found in {source_path} or next to {__file__}."
        )

    from huggingface_hub import hf_hub_download

    repo_id = E1_TOKENIZER_REPO_ID
    if source_path is not None and not checked_local_source:
        repo_id = source_path
    try:
        fname = hf_hub_download(
            repo_id=repo_id,
            filename="tokenizer.json",
            cache_dir=os.fspath(cache_dir) if cache_dir is not None else None,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
    except Exception as error:
        raise FileNotFoundError(
            f"E1 tokenizer.json was not found locally and could not be loaded from {repo_id}."
        ) from error
    return _load_tokenizer_file(fname)


@dataclass
class DataPrepConfig:
    max_num_sequences: int = 512
    max_num_positions_within_seq: int = 8192
    remove_X_tokens: bool = False


def get_context(sequence: str) -> str | None:
    if "," in sequence:
        return sequence.rsplit(",", 1)[0]
    return None


class E1BatchPreparer:
    def __init__(
        self,
        data_prep_config: DataPrepConfig | None = None,
        tokenizer: Tokenizer | None = None,
        tokenizer_source: str | os.PathLike | None = None,
        local_files_only: bool = False,
        cache_dir: str | os.PathLike | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        preserve_context_labels: bool = False,
    ):
        self.tokenizer = tokenizer or get_tokenizer(
            tokenizer_source,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
        )
        self.data_prep_config = data_prep_config or DataPrepConfig()
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        self.preserve_context_labels = preserve_context_labels
        self.boundary_token_ids = torch.tensor(
            [self.tokenizer.token_to_id(token) for token in ["<bos>", "<eos>", "1", "2", "<pad>"]]
        ).long()
        self.mask_token = "?"  # nosec
        self.mask_token_id = self.tokenizer.token_to_id(self.mask_token)
        self.X_token_id = self.tokenizer.token_to_id("X")
        self.vocab = self.tokenizer.get_vocab()

    def get_batch_kwargs(  # type: ignore[override]
        self,
        sequences: list[str],
        device: torch.device | None = None,
        non_blocking: bool = False,
    ) -> dict[str, torch.Tensor | list[str] | list[int]]:
        device = torch.device("cpu") if device is None else device
        sequence_encodings = [self.prepare_multiseq(sequence) for sequence in sequences]
        return self.pad_encodings(sequence_encodings, device, non_blocking)

    def pad_encodings(
        self,
        sequence_encodings: list[dict[str, torch.Tensor]],
        device: torch.device | None = None,
        non_blocking: bool = False,
    ) -> dict[str, torch.Tensor | list[str] | list[int]]:
        device = torch.device("cpu") if device is None else device
        non_blocking = non_blocking and device.type == "cuda"
        padded_encodings = {}
        # Note: We use -1 as the padding value for sequence and position ids because the 0 value
        # is a valid value for sequence and position ids. -1 is then used to distinguish valid
        # tokens from padding tokens, for example, when doing padding/unpadding for flash attention.
        for key, padding_value in {
            "input_ids": self.pad_token_id,
            "sequence_ids": -1,
            "within_seq_position_ids": -1,
            "global_position_ids": -1,
            "labels": self.pad_token_id,
        }.items():
            padded_encodings[key] = pad_sequence(
                [enc[key] for enc in sequence_encodings],
                batch_first=True,
                padding_value=padding_value,
            ).to(device=device, dtype=torch.long, non_blocking=non_blocking)

        padded_encodings["context"] = [enc["context"] for enc in sequence_encodings]
        padded_encodings["context_len"] = [enc["context_len"] for enc in sequence_encodings]

        return padded_encodings

    def prepare_multiseq(self, sequence: str) -> dict[str, torch.Tensor | str | int]:
        sequences = sequence.split(",")
        if len(sequences) > self.data_prep_config.max_num_sequences:
            raise ValueError(
                f"Number of sequences {len(sequences)} exceeds max number of sequences "
                f"{self.data_prep_config.max_num_sequences} in the provided multi-sequence "
                "instance. Please remove some homologous sequences before trying again."
            )

        encodings = tuple(self.prepare_singleseq(item) for item in sequences)
        token_counts = torch.tensor(
            [encoding["input_ids"].numel() for encoding in encodings],
            dtype=torch.long,
        )
        input_ids = torch.cat(tuple(encoding["input_ids"] for encoding in encodings))
        labels = torch.cat(tuple(encoding["labels"] for encoding in encodings))
        positions = tuple(encoding["position_ids"] for encoding in encodings)
        within_seq_position_ids = torch.cat(positions)

        # Offsets preserve gaps left by optional X-token removal.
        position_spans = torch.tensor(
            [int(position_ids[-1].item()) + 1 for position_ids in positions],
            dtype=torch.long,
        )
        position_offsets = torch.cat(
            (torch.zeros(1, dtype=torch.long), position_spans[:-1]),
        ).cumsum(dim=0)
        global_position_ids = torch.cat(
            tuple(
                position_ids + offset
                for position_ids, offset in zip(positions, position_offsets, strict=True)
            )
        )
        sequence_ids = torch.arange(len(encodings), dtype=torch.long).repeat_interleave(
            token_counts
        )

        context_len = int(token_counts[:-1].sum().item())
        context = self.tokenizer.decode(input_ids[:context_len].tolist(), skip_special_tokens=False)
        if not self.preserve_context_labels:
            labels[:context_len] = self.pad_token_id

        aligned_tensors = (
            sequence_ids,
            within_seq_position_ids,
            global_position_ids,
            labels,
        )
        if any(tensor.shape != input_ids.shape for tensor in aligned_tensors):
            raise AssertionError(
                "Input ids, sequence ids, within seq position ids, global position ids, "
                "and labels must have the same shape"
            )
        if input_ids.numel() < context_len:
            raise AssertionError(
                "Input ids must have at least as many tokens as the context length"
            )

        return {
            "input_ids": input_ids,
            "sequence_ids": sequence_ids,
            "within_seq_position_ids": within_seq_position_ids,
            "global_position_ids": global_position_ids,
            "labels": labels,
            "context": context,
            "context_len": context_len,
        }

    def prepare_singleseq(self, sequence: str) -> dict[str, torch.Tensor]:
        if not self.validate_sequence(sequence):
            raise ValueError(
                f"Invalid sequence: {sequence}; Input sequence should contain "
                "[A-Z] or ? characters only"
            )

        if len(sequence) > self.data_prep_config.max_num_positions_within_seq:
            raise ValueError(
                f"Sequence length {len(sequence)} exceeds max length "
                f"{self.data_prep_config.max_num_positions_within_seq}"
            )

        symbols = itertools.chain(("<bos>", "1"), sequence, ("2", "<eos>"))
        tokens = torch.tensor(
            [self.vocab[symbol] for symbol in symbols],
            dtype=torch.long,
        )
        position_ids = torch.arange(tokens.numel(), dtype=torch.long)

        if self.data_prep_config.remove_X_tokens:
            keep = tokens.ne(self.X_token_id)
            tokens = tokens[keep]
            position_ids = position_ids[keep]

        return {"input_ids": tokens, "labels": tokens, "position_ids": position_ids}

    def get_boundary_token_mask(self, tokens: torch.Tensor) -> torch.BoolTensor:
        return torch.isin(tokens, self.boundary_token_ids.to(tokens.device))

    def get_mask_positions_mask(self, tokens: torch.Tensor) -> torch.BoolTensor:
        return tokens == self.mask_token_id

    def validate_sequence(self, sequence: str) -> bool:
        assert isinstance(sequence, str), "Sequence must be a string"
        sequence = sequence.replace(self.mask_token, "")
        return sequence.isalpha() and sequence.isupper()
