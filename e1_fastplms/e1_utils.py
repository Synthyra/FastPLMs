# io.py:
import biotite.sequence.io.fasta as fasta


def read_fasta_sequences(path: str) -> dict[str, str]:
    sequences: dict[str, str] = {}
    with open(path, "r") as f:
        for header, sequence in fasta.FastaFile.read(f).items():
            sequences[header] = sequence
    return sequences


def write_fasta_sequences(path: str, sequences: dict[str, str]) -> None:
    with open(path, "w") as f:
        for header, sequence in sequences.items():
            f.write(f">{header}\n{sequence}\n")

# msa_sampling.py:

import os
import string
import tempfile
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import torch

LOWERCASE_CHARS = string.ascii_lowercase.encode("ascii")

IdSequence = namedtuple("IdSequence", ["id", "sequence"])


@dataclass
class ContextSpecification:
    max_num_samples: int = 511
    max_token_length: int = 32768
    max_query_similarity: float = 1.0
    min_query_similarity: float = 0.0
    neighbor_similarity_lower_bound: float = 0.8


def parse_msa(path: str) -> list[IdSequence]:
    """
    Parse a MSA file in a3m format. Convert any . dot to - character.

    Args:
        path: Path to the MSA file.

    Returns:
        list[IdSequence]: A list of IdSequence objects.
    """
    records = list(read_fasta_sequences(path).items())
    sequences = []
    for record_id, record_seq in records:
        sequences.append(IdSequence(record_id, str(record_seq).replace("\x00", "").replace(".", "-")))
    return sequences


def convert_to_tensor(sequences: list[IdSequence], device: torch.device | None = None) -> torch.ByteTensor:
    """
    Convert MSA Sequences to a Byte Tensor. Remove any lowercase characters which represent indels.
    Move the tensor to the specified device.

    Args:
        sequences: List of IdSequence objects.
        device: Device to move the tensor to. If None, picks cuda if available, else cpu.

    Returns:
        torch.ByteTensor: A byte tensor of the MSA sequences.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    byte_seqs = [x.sequence.encode("ascii").translate(None, LOWERCASE_CHARS) for x in sequences]
    byte_seqs = np.vstack([np.frombuffer(x, dtype=np.uint8) for x in byte_seqs])
    byte_seqs = torch.from_numpy(byte_seqs)
    byte_seqs = byte_seqs.to(device)

    return byte_seqs


def get_num_neighbors(byte_seqs: torch.ByteTensor, sim_threshold: float = 0.8) -> list[int]:
    """
    Get the number of neighbors for each sequence in the MSA. A neighbor to a sequence is another sequence that
    is atleast sim_threshold similar to the former sequence. Similarity is calculated as the fraction of
    non-gap characters that are the same between the two sequences.

    This is used to sample the context sequences from the MSA (with weight 1/number of neighbors).

    Args:
        byte_seqs: Byte tensor of the MSA sequences.
        sim_threshold: Similarity threshold for a sequence to be considered a neighbor.

    Returns:
        list[int]: A list of the number of neighbors for each sequence in the MSA.
    """
    gap_token_id = np.frombuffer(b"-", np.uint8)[0].item()
    seq_lens = (byte_seqs != gap_token_id).sum(dim=1)
    num_neighbors = []
    for i in range(byte_seqs.shape[0]):
        query_non_gaps = byte_seqs[i] != gap_token_id
        seqs_sim = (byte_seqs[:, query_non_gaps] == byte_seqs[i, query_non_gaps]).sum(dim=1) / seq_lens
        num_neighbors.append((seqs_sim >= sim_threshold).sum().item())
    return num_neighbors


def get_similarity_to_query(byte_seqs: torch.ByteTensor) -> torch.FloatTensor:
    """
    Get the similarity of each sequence in the MSA to the query sequence (the first sequence in the MSA).
    Similarity is calculated as the fraction of characters that are the same between the two sequences.

    Args:
        byte_seqs: Byte tensor of the MSA sequences.

    Returns:
        torch.FloatTensor: A float tensor of the similarity of each sequence in the MSA to the query sequence.
    """
    return (byte_seqs == byte_seqs[0, :]).sum(dim=1) / byte_seqs.shape[1]


def sample_context(
    msa_path: str,
    max_num_samples: int,
    max_token_length: int,
    max_query_similarity: float = 1.0,
    min_query_similarity: float = 0.0,
    neighbor_similarity_lower_bound: float = 0.8,
    use_full_sequences_in_context: bool = False,
    full_sequences_path: str | None = None,
    seed: int = 0,
    device: torch.device | None = None,
    cache_num_neighbors_path: str | None = None,
) -> tuple[str, list[str]]:
    """
    Sample a context from a given MSA to use with E1/Progen3-RA Models.

    Args:
        msa_path: Path to the MSA file. The first item in MSA file should be the query sequence.
            The msa should be a A3M file.
        max_num_samples: Maximum number of samples to draw from the MSA.
        max_token_length: Maximum length of the context in tokens.
        max_query_similarity: Maximum similarity of context sequences to the query sequence.
        min_query_similarity: Minimum similarity of context sequences to the query sequence.
        neighbor_similarity_lower_bound: Minimum similarity to be considered a neighbor.
        use_full_sequences_in_context: Whether to use full sequences in the context
            or the aligned fragments from the MSA.
        full_sequences_path: Path to the full sequences file. Should contain same number of sequences
            as the MSA file and same ids as the MSA file.
        seed: Random seed.
        device: Device to use for the computation. If None, picks cuda if available, else cpu.
        cache_num_neighbors_path: Path to cache the number of neighbors computed for each sequence in MSA.
            Should be a npy file.

    Returns:
        tuple[str, list[str]]: A tuple containing the context string (sequences concatenated with commans)
        and the ids of the sequences in the context.

    Asserts:
        AssertionError: If the number of full sequences does not match the number of MSA sequences.
        AssertionError: If the ids of the full sequences and MSA sequences do not match.
        AssertionError: If the number of samples is less than 1.
    """
    msa_sequences = parse_msa(msa_path)
    msa_as_byte_tensor = convert_to_tensor(msa_sequences, device)
    if cache_num_neighbors_path is not None and os.path.exists(cache_num_neighbors_path):
        num_neighbors = np.load(cache_num_neighbors_path)
    else:
        num_neighbors = get_num_neighbors(msa_as_byte_tensor, neighbor_similarity_lower_bound)
        num_neighbors = np.array(num_neighbors)

        if cache_num_neighbors_path is not None:
            np.save(cache_num_neighbors_path, num_neighbors)

    sampling_weights = 1.0 / num_neighbors
    query_similarity = get_similarity_to_query(msa_as_byte_tensor)
    filtered_indices = (query_similarity <= max_query_similarity) & (query_similarity >= min_query_similarity)

    assert filtered_indices.sum() >= 1, (
        f"No sequences found with similarity to query within the given range: {min_query_similarity=} <= query_similarity <= {max_query_similarity=}. "
        "Consider increasing the max_query_similarity or decreasing the min_query_similarity."
    )
    filtered_weights = sampling_weights[filtered_indices.cpu().numpy()]

    sampled_indices = np.random.default_rng(seed).choice(
        len(filtered_weights),
        size=min(max_num_samples, len(filtered_weights)),
        p=filtered_weights / filtered_weights.sum(),
        replace=False,
        shuffle=True,
    )

    if use_full_sequences_in_context:
        assert full_sequences_path is not None
        full_sequences = parse_msa(full_sequences_path)
        assert len(full_sequences) == len(msa_sequences), "Number of full sequences must match number of MSA sequences"
        for i, (full_seq, msa_seq) in enumerate(zip(full_sequences, msa_sequences)):
            assert full_seq.id == msa_seq.id, (
                "Full sequences and MSA sequences should be in the same order in the files and have same ids. "
                f"Found differing id for sample {i}: {full_seq.id} != {msa_seq.id}"
            )

        sampled_sequences = [full_sequences[i] for i in sampled_indices]
    else:
        sampled_sequences = [msa_sequences[i] for i in sampled_indices]

    context_sequences = []
    context_ids = []
    context_length = 0
    for seq in sampled_sequences:
        seq_str = seq.sequence.upper().encode("ascii").translate(None, b"-").decode("ascii")
        if context_length + len(seq_str) > max_token_length:
            break
        context_sequences.append(seq_str)
        context_ids.append(seq.id)
        context_length += len(seq_str)

    return ",".join(context_sequences), context_ids


def sample_multiple_contexts(
    msa_path: str,
    context_specifications: list[ContextSpecification],
    use_full_sequences_in_context: bool = False,
    full_sequences_path: str | None = None,
    seed: int = 0,
    device: torch.device | None = None,
    cache_num_neighbors_path: str | None = None,
) -> tuple[list[str], list[list[str]]]:
    """
    Sample multiple contexts from a given MSA to use with E1/Progen3-RA Models with different
    context specifications that we can ensemble over eventually. A context specification is an instance of the ContextSpecification class
    and specifies the maximum number of samples, maximum token length, maximum query similarity,
    minimum query similarity, and neighbor similarity lower bound.

    Args:
        msa_path: Path to the MSA file. The first item in MSA file should be the query sequence.
            The msa should be a A3M file.
        context_specifications: List of context specifications to sample from the MSA.
        use_full_sequences_in_context: Whether to use full sequences in the context
            or the aligned fragments from the MSA.
        full_sequences_path: Path to the full sequences file. Should contain same number of sequences
            as the MSA file and same ids as the MSA file.
        seed: Random seed.
        device: Device to use for the computation. If None, picks cuda if available, else cpu.
        cache_num_neighbors_path: Path to cache the number of neighbors computed for each sequence in MSA.
            Should be a npy file. If None, a temporary file will be created and deleted after the computation.

    Returns:
        tuple[list[str], list[list[str]]]: A tuple containing the list of context strings and the list of
        list of ids of the sequences in each context.

    Asserts:
        AssertionError: If the number of full sequences does not match the number of MSA sequences.
        AssertionError: If the ids of the full sequences and MSA sequences do not match.
        AssertionError: If the number of samples is less than 1.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        if cache_num_neighbors_path is None:
            cache_num_neighbors_path = os.path.join(temp_dir, "num_neighbors.npy")

        contexts = []
        context_ids = []
        for i, context_specification in enumerate(context_specifications):
            context, ids = sample_context(
                msa_path=msa_path,
                max_num_samples=context_specification.max_num_samples,
                max_token_length=context_specification.max_token_length,
                max_query_similarity=context_specification.max_query_similarity,
                min_query_similarity=context_specification.min_query_similarity,
                neighbor_similarity_lower_bound=context_specification.neighbor_similarity_lower_bound,
                use_full_sequences_in_context=use_full_sequences_in_context,
                full_sequences_path=full_sequences_path,
                seed=seed + i,
                device=device,
                cache_num_neighbors_path=cache_num_neighbors_path,
            )
            contexts.append(context)
            context_ids.append(ids)

    return contexts, context_ids

# predictor.py:

import itertools
import logging
from collections import defaultdict
from collections.abc import Sequence
from typing import Iterator, TypedDict
import torch
from tqdm import tqdm
from .modeling_e1 import E1ForMaskedLM, E1MaskedLMOutputWithPast, E1BatchPreparer, get_context, DataPrepConfig, KVCache

IndexedSequence = tuple[int, str]
logger = logging.getLogger(__name__)


class E1Prediction(TypedDict, total=False):
    id: str | int
    context_id: str | int | None
    logits: torch.Tensor
    token_embeddings: torch.Tensor
    mean_token_embeddings: torch.Tensor


class E1Predictor:
    def __init__(
        self,
        model: E1ForMaskedLM,
        data_prep_config: DataPrepConfig | None = None,
        max_batch_tokens: int = 65536,
        use_cache: bool = True,
        cache_size: int = 4,
        save_masked_positions_only: bool = False,
        fields_to_save: list[str] = ["logits", "token_embeddings", "mean_token_embeddings"],
        keep_predictions_in_gpu: bool = False,
    ):
        self.model = model
        self.max_batch_tokens = max_batch_tokens
        self.batch_preparer = E1BatchPreparer(data_prep_config=data_prep_config)
        self.model.eval()
        self.kv_cache = KVCache(cache_size=cache_size) if use_cache else None

        self.fields_to_save = fields_to_save
        self.save_masked_positions_only = save_masked_positions_only
        self.keep_predictions_in_gpu = keep_predictions_in_gpu

    def group_by_length(self, indexed_sequences: list[IndexedSequence]) -> list[list[IndexedSequence]]:
        batches: list[list[IndexedSequence]] = [[]]
        for idx, seq in sorted(indexed_sequences, key=lambda idx_seq: (len(idx_seq[1]), idx_seq[0])):
            if len(batches[-1]) > 0 and len(seq) * (len(batches[-1]) + 1) > self.max_batch_tokens:
                batches.append([])
            batches[-1].append((idx, seq))

        return batches

    def group_by_context(self, indexed_sequences: list[IndexedSequence]) -> list[list[IndexedSequence]]:
        batches: dict[str | None, list[IndexedSequence]] = defaultdict(list)
        for idx, seq in indexed_sequences:
            batches[get_context(seq)].append((idx, seq))
        return list(batches.values())

    def batch_sequences(self, sequences: list[str]) -> list[tuple[list[int], bool]]:  # type: ignore[override]
        """
        Batches the sequences and returns indices for the current rank
        We want to keep sequences of similar length together.
        Ensures that no batch exceeds max_batch_tokens
        [For E1, also ensures if context is present, preserve locality of context]
        """
        indexed_sequences: list[IndexedSequence] = list(enumerate(sequences))
        indexed_batches = self.group_by_context(indexed_sequences)
        # Preserve context ordering
        indexed_batches = list(
            itertools.chain.from_iterable([self.group_by_length(batch) for batch in indexed_batches])
        )
        batches = [[item[0] for item in batch] for batch in indexed_batches]  # type: ignore[no-redef,misc]

        assert sorted(sum(batches, [])) == list(range(len(sequences))), (
            "Batches must contain all indices with no repetition"
        )

        batches_with_validity = [(b, True) for b in batches]

        return batches_with_validity

    @torch.no_grad()
    def predict_batch(self, sequences: list[str], sequence_metadata: list[dict[str, str | int]]) -> list[E1Prediction]:
        """
        Returns the logits/embeddings for the last sequence for multi-sequence inputs.
        """
        outputs = self.predict_batch_padded(sequences)
        outputs["logits"] = outputs["logits"].float()
        outputs["embeddings"] = outputs["embeddings"].float()

        token_mask = outputs["non_boundary_token_mask"] & outputs["last_sequence_mask"]
        if self.save_masked_positions_only:
            token_mask = token_mask & outputs["mask_positions_mask"]
        predictions = []
        for i in range(len(sequences)):
            pred: E1Prediction = {
                "id": sequence_metadata[i]["id"],
                "context_id": sequence_metadata[i].get("context_id", None),
            }
            if "logits" in self.fields_to_save:
                pred["logits"] = outputs["logits"][i, token_mask[i]]
                if not self.keep_predictions_in_gpu:
                    pred["logits"] = pred["logits"].to("cpu")  # type: ignore[union-attr]
            if "token_embeddings" in self.fields_to_save:
                pred["token_embeddings"] = outputs["embeddings"][i, token_mask[i]]
                if not self.keep_predictions_in_gpu:
                    pred["token_embeddings"] = pred["token_embeddings"].to("cpu")  # type: ignore[union-attr]
            if "mean_token_embeddings" in self.fields_to_save:
                pred["mean_token_embeddings"] = outputs["embeddings"][i, token_mask[i]].mean(dim=0)
                if not self.keep_predictions_in_gpu:
                    pred["mean_token_embeddings"] = pred["mean_token_embeddings"].to("cpu")  # type: ignore[union-attr]
            predictions.append(pred)
        return predictions

    @torch.no_grad()
    def predict_batch_padded(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        """
        If use_cache is True, this function will return the logits/embeddings for the last sequence for multi-sequence inputs.
        If use_cache is False, this function will return the logits/embeddings for every sequence for multi-sequence inputs.

        Returns three additional masks:
        - non_boundary_token_mask: True for tokens that are part of the input sequence i.e not boundary tokens like 1, 2, <bos>, <eos>, <pad>, etc.
        - last_sequence_mask: True for tokens that are part of the last sequence (including boundary tokens) in case of multi-sequence input.
        - mask_positions_mask: True for masked positions.
        - valid_token_mask: True for valid tokens.
        """
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type, torch.bfloat16):
            batch = self.batch_preparer.get_batch_kwargs(sequences, device=torch.device(device_type))

            if self.kv_cache is not None:
                self.kv_cache.before_forward(batch)

            output: E1MaskedLMOutputWithPast = self.model(
                input_ids=batch["input_ids"],
                within_seq_position_ids=batch["within_seq_position_ids"],
                global_position_ids=batch["global_position_ids"],
                sequence_ids=batch["sequence_ids"],
                past_key_values=batch.get("past_key_values", None),
                use_cache=batch.get("use_cache", False),
                output_attentions=False,
                output_hidden_states=False,
            )
            if self.kv_cache is not None:
                self.kv_cache.after_forward(batch, output)

            logits = output.logits
            embeddings = output.last_hidden_state

            padding_mask = batch["input_ids"] == self.batch_preparer.pad_token_id
            last_sequence_mask = batch["sequence_ids"] == batch["sequence_ids"].max(dim=1)[0][:, None]  # type: ignore[union-attr]
            boundary_token_mask = self.batch_preparer.get_boundary_token_mask(batch["input_ids"])
            mask_positions_mask = self.batch_preparer.get_mask_positions_mask(batch["input_ids"])

            return {
                "logits": logits,
                "embeddings": embeddings,
                "last_sequence_mask": last_sequence_mask,
                "non_boundary_token_mask": ~boundary_token_mask,
                "mask_positions_mask": mask_positions_mask,
                "valid_token_mask": ~padding_mask,
            }

    @torch.no_grad()
    def predict(
        self,
        sequences: Sequence[str],
        sequence_ids: Sequence[int | str] | None = None,
        context_seqs: dict[str, str] | None = None,
    ) -> Iterator[E1Prediction]:
        if sequence_ids is None:
            sequence_ids = list(range(len(sequences)))
        if context_seqs:
            sequences_with_context = [
                (ctx + "," + seq, {"context_id": ctx_id, "id": sequence_id})
                for ctx_id, ctx in context_seqs.items()
                for seq, sequence_id in zip(sequences, sequence_ids)
            ]
        else:
            sequences_with_context = [(seq, {"id": sequence_id}) for seq, sequence_id in zip(sequences, sequence_ids)]
        sequences, sequence_metadata = tuple(zip(*sequences_with_context))  # type: ignore[assignment]
        sequence_batch_indices: list[tuple[list[int], bool]] = self.batch_sequences(sequences)  # type: ignore[arg-type]
        logger.info(f"Predicting for {len(sequence_batch_indices)} batches")

        for indices, is_valid_batch in tqdm(
            sequence_batch_indices, desc="Predicting batches"
        ):
            sequence_batch = [sequences[i] for i in indices]
            sequence_batch_metadata = [sequence_metadata[i] for i in indices]
            batch_predictions = self.predict_batch(sequence_batch, sequence_batch_metadata)

            if not is_valid_batch:
                continue

            for prediction in batch_predictions:
                yield prediction