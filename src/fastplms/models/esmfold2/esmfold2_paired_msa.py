"""Construct taxonomy-paired MSA features for multichain folding."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .esmfold2_constants import (
    MSA_GAP_TOKEN_ID,
    PROTEIN_3TO1,
    PROTEIN_RESIDUE_TO_RES_TYPE,
    PROTEIN_UNK_RES_TYPE,
)
from .esmfold2_msa import MSA

_TAXONOMY_PATTERN = re.compile(r"key=(-?\d+)")


def protein_letter_to_res_type() -> dict[str, int]:
    """Return the one-letter residue vocabulary used by the MSA encoder."""

    vocabulary = {
        one_letter: PROTEIN_RESIDUE_TO_RES_TYPE[three_letter]
        for three_letter, one_letter in PROTEIN_3TO1.items()
        if three_letter in PROTEIN_RESIDUE_TO_RES_TYPE
    }
    vocabulary.update({"-": MSA_GAP_TOKEN_ID, "X": PROTEIN_UNK_RES_TYPE})
    return vocabulary


def _taxonomy_from_header(header: str) -> int:
    match = _TAXONOMY_PATTERN.search(header) if header else None
    return int(match.group(1)) if match is not None else -1


def _emitted_length(sequence: str) -> int:
    return sum(character != "." and not character.islower() for character in sequence)


def _decode_a3m_row(
    sequence: str,
    sequence_length: int,
    vocabulary: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    residues = np.full(sequence_length, MSA_GAP_TOKEN_ID, dtype=np.int64)
    deletions = np.zeros(sequence_length, dtype=np.float32)
    column = 0
    insertion_count = 0
    for character in sequence:
        if character == "." or character.islower():
            insertion_count += 1
            continue
        if column == sequence_length:
            break
        residues[column] = (
            MSA_GAP_TOKEN_ID
            if character == "-"
            else vocabulary.get(character.upper(), PROTEIN_UNK_RES_TYPE)
        )
        if insertion_count:
            deletions[column] = float(insertion_count)
            insertion_count = 0
        column += 1
    return residues, deletions


def msa_to_res_type_and_deletions(
    msa: MSA,
    letter_to_res_type: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Decode an A3M alignment into arrays ``X`` and ``D`` with shape (m, l)."""

    sequence_length = _emitted_length(msa.entries[0].sequence)
    residue_rows: list[np.ndarray] = []
    deletion_rows: list[np.ndarray] = []
    for entry in msa.entries:
        residues, deletions = _decode_a3m_row(
            entry.sequence,
            sequence_length,
            letter_to_res_type,
        )
        residue_rows.append(residues)
        deletion_rows.append(deletions)
    return np.stack(residue_rows), np.stack(deletion_rows)


@dataclass(frozen=True)
class _ChainAlignment:
    residues: np.ndarray
    deletions: np.ndarray
    taxonomies: list[int]


def _chain_alignment(
    msa: MSA | None,
    query_res_types: np.ndarray,
    vocabulary: dict[str, int],
) -> _ChainAlignment:
    if msa is None or msa.depth == 0:
        return _ChainAlignment(
            residues=query_res_types[None, :],
            deletions=np.zeros((1, query_res_types.shape[0]), dtype=np.float32),
            taxonomies=[-1],
        )
    residues, deletions = msa_to_res_type_and_deletions(msa, vocabulary)
    taxonomies = [_taxonomy_from_header(entry.header) for entry in msa.entries]
    return _ChainAlignment(residues, deletions, taxonomies)


def _taxonomy_groups(
    chain_ids: list[int],
    alignments: dict[int, _ChainAlignment],
) -> dict[int, list[tuple[int, int]]]:
    groups: dict[int, list[tuple[int, int]]] = {}
    for chain_id in chain_ids:
        for row, taxonomy in enumerate(alignments[chain_id].taxonomies):
            if row and taxonomy != -1:
                groups.setdefault(taxonomy, []).append((chain_id, row))
    return {taxonomy: rows for taxonomy, rows in groups.items() if len(rows) > 1}


def _available_rows(
    chain_ids: list[int],
    alignments: dict[int, _ChainAlignment],
    groups: dict[int, list[tuple[int, int]]],
) -> dict[int, list[int]]:
    used = {row for group in groups.values() for row in group}
    return {
        chain_id: [
            row
            for row in range(1, len(alignments[chain_id].taxonomies))
            if (chain_id, row) not in used
        ]
        for chain_id in chain_ids
    }


def _append_taxonomy_rows(
    rows: list[dict[int, int]],
    paired_flags: list[dict[int, int]],
    chain_ids: list[int],
    groups: dict[int, list[tuple[int, int]]],
    available: dict[int, list[int]],
    max_pairs: int,
) -> None:
    ordered_groups = sorted(
        groups.values(),
        key=lambda group: len({chain_id for chain_id, _row in group}),
        reverse=True,
    )
    for group in ordered_groups:
        rows_by_chain: dict[int, list[int]] = {}
        for chain_id, row in group:
            rows_by_chain.setdefault(chain_id, []).append(row)
        for occurrence in range(max(map(len, rows_by_chain.values()))):
            selected: dict[int, int] = {}
            flags: dict[int, int] = {}
            for chain_id, candidates in rows_by_chain.items():
                selected[chain_id] = candidates[occurrence % len(candidates)]
                flags[chain_id] = 1
            for chain_id in chain_ids:
                if chain_id not in selected:
                    flags[chain_id] = 0
                    selected[chain_id] = available[chain_id].pop(0) if available[chain_id] else -1
            rows.append(selected)
            paired_flags.append(flags)
            if len(rows) >= max_pairs:
                break
        if len(rows) >= max_pairs:
            break


def _append_unpaired_rows(
    rows: list[dict[int, int]],
    paired_flags: list[dict[int, int]],
    chain_ids: list[int],
    available: dict[int, list[int]],
    max_total: int,
) -> None:
    max_remaining = max((len(indices) for indices in available.values()), default=0)
    for _ in range(min(max_total - len(rows), max_remaining)):
        rows.append(
            {
                chain_id: available[chain_id].pop(0) if available[chain_id] else -1
                for chain_id in chain_ids
            }
        )
        paired_flags.append({chain_id: 0 for chain_id in chain_ids})
        if len(rows) >= max_total:
            break


def _pairing_plan(
    chain_ids: list[int],
    alignments: dict[int, _ChainAlignment],
    max_pairs: int,
    max_total: int,
    max_seqs: int,
) -> tuple[list[dict[int, int]], list[dict[int, int]]]:
    groups = _taxonomy_groups(chain_ids, alignments)
    available = _available_rows(chain_ids, alignments, groups)
    rows = [{chain_id: 0 for chain_id in chain_ids}]
    flags = [{chain_id: 1 for chain_id in chain_ids}]
    _append_taxonomy_rows(rows, flags, chain_ids, groups, available, max_pairs)
    _append_unpaired_rows(rows, flags, chain_ids, available, max_total)
    return rows[:max_seqs], flags[:max_seqs]


def _project_alignment_rows(
    chain_ids: list[int],
    alignments: dict[int, _ChainAlignment],
    rows: list[dict[int, int]],
    flags: list[dict[int, int]],
    token_asym_ids: np.ndarray,
    token_res_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m, t = len(rows), len(token_asym_ids)
    residues = np.full((m, t), MSA_GAP_TOKEN_ID, dtype=np.int64)
    deletions = np.zeros((m, t), dtype=np.float32)
    paired_mask = np.zeros((m, t), dtype=np.float32)
    for chain_id in chain_ids:
        alignment = alignments[chain_id]
        selected_rows = np.asarray([row[chain_id] for row in rows], dtype=np.int64)
        chain_flags = np.asarray([row[chain_id] for row in flags], dtype=np.float32)
        token_mask = token_asym_ids == chain_id
        if not token_mask.any():
            continue
        columns = np.minimum(token_res_ids[token_mask], alignment.residues.shape[1] - 1)
        valid_rows = selected_rows >= 0
        if valid_rows.any():
            output_rows = np.flatnonzero(valid_rows)
            output_columns = np.flatnonzero(token_mask)
            residues[np.ix_(output_rows, output_columns)] = alignment.residues[
                selected_rows[valid_rows]
            ][:, columns]
            deletions[np.ix_(output_rows, output_columns)] = alignment.deletions[
                selected_rows[valid_rows]
            ][:, columns]
        paired_mask[:, token_mask] = chain_flags[:, None]
    return residues, deletions, paired_mask


def construct_paired_msa(
    chain_msas: dict[int, MSA | None],
    chain_query_res_types: dict[int, np.ndarray],
    token_asym_ids: np.ndarray,
    token_res_ids: np.ndarray,
    letter_to_res_type: dict[str, int] | None = None,
    *,
    max_pairs: int = 8192,
    max_total: int = 16384,
    max_seqs: int = 16384,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return residue, deletion, and pairing arrays with shape (m, t)."""

    vocabulary = protein_letter_to_res_type() if letter_to_res_type is None else letter_to_res_type
    chain_ids = sorted(chain_msas)
    alignments = {
        chain_id: _chain_alignment(
            chain_msas[chain_id],
            chain_query_res_types[chain_id],
            vocabulary,
        )
        for chain_id in chain_ids
    }
    rows, flags = _pairing_plan(
        chain_ids,
        alignments,
        max_pairs,
        max_total,
        max_seqs,
    )
    return _project_alignment_rows(
        chain_ids,
        alignments,
        rows,
        flags,
        token_asym_ids,
        token_res_ids,
    )
