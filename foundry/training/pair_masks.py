"""Off-diagonal negative masks for paired batches, applied from tables built once per release.

A paired batch holds `b` positive rows. Cell `(i, j)` of its `(b, b)` score matrix pairs the left protein of row `i` with the
right protein of row `j`, so the diagonal is the positives and every other cell is a candidate negative. Three checks cut the
candidates, in this order (docs/conventions/paired_sampling.md):

1. taxonomy: a cell is a candidate only inside its row's ordered taxonomy pair;
2. homology: a candidate is removed when its two proteins sit in the clusters of a known positive's two proteins, in either
   orientation, because it is then a likely false negative;
3. metadata: the surviving candidates are thinned per stratum, so that no property of a pair that is not the label separates
   them from the positives.

A known positive pair is a positive of the loss, never removed, and a protein paired with itself is never a negative.

Every check is a gather or a `searchsorted` over tables computed offline (`MaskTables`): no alignment and no Python loop per
batch. The only host work is one dictionary lookup per row to turn protein ids into table indices.

Shape symbols: `b` rows in a batch, so `(b, b)` cells; `n_p` proteins; `n_c` clusters; `n_k` known positive cluster pairs;
`n_e` known positive protein pairs; `n_v` GO-BP vocabulary terms; `w` GO terms kept per protein; `n_s` strata.
"""

from __future__ import annotations

import numpy as np
import torch

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple
from torch import Tensor

from foundry.digests import sha256_file


TABLE_SCHEMA = "pair_mask_tables_v1"
# GO-BP Jaccard overlap bins: none, up to 0.1, up to 0.25, up to 0.5, and above.
JACCARD_EDGES = (0.0, 0.1, 0.25, 0.5)
OVERLAP_BINS = len(JACCARD_EDGES) + 1
# The gap between two proteins' log2 length bins, 0 to 3 or more.
LENGTH_GAPS = 4
# A stratum is (overlap bin, shares a location, shares a source, length gap), in mixed radix.
STRATA = OVERLAP_BINS * 2 * 2 * LENGTH_GAPS
# Lengths below 64 fall in bin 0, then one bin per doubling up to 1024 and above.
LENGTH_BINS = 6
FIRST_BIN_LENGTH = 64


class MaskTables(NamedTuple):
    """Everything a mask reads of a release, as arrays indexed by protein: built once, saved, and loaded onto the device.

    `species`, `cluster` and the metadata arrays hold one entry per protein of `proteins`, which are sorted. A cluster pair key is
    `min * n_clusters + max` over the two cluster ids, and a protein pair key is `min * n_p + max` over the two protein indices,
    both sorted and unique, so one key covers a pair in either orientation. `go_terms` lists each protein's GO-BP vocabulary ids,
    `-1` padding the unused entries. `keep_fraction` is the share of each stratum's candidates a batch keeps.
    """

    proteins: tuple[str, ...]
    species: np.ndarray  # (n_p,) int64, taxonomy id each protein is named by
    cluster: np.ndarray  # (n_p,) int64, dense id of the mask-identity cluster
    n_clusters: int
    cluster_pair_keys: np.ndarray  # (n_k,) int64, sorted and unique
    exact_pair_keys: np.ndarray  # (n_e,) int64, sorted and unique
    source_mask: np.ndarray  # (n_p,) uint8, one bit per source of the protein's known positives
    length_bin: np.ndarray  # (n_p,) uint8, in [0, LENGTH_BINS)
    location_mask: np.ndarray  # (n_p,) int64, one bit per subcellular location, 0 when unknown
    go_terms: np.ndarray  # (n_p, w) int16, GO-BP vocabulary ids, -1 padded
    n_go_terms: int
    keep_fraction: np.ndarray  # (n_s,) float32, in [0, 1]

    def checked(self) -> MaskTables:
        """The tables, after refusing arrays whose shape, dtype, order or range breaks what the mask assumes."""
        n_p = len(self.proteins)
        per_protein = {"species": np.int64, "cluster": np.int64, "source_mask": np.uint8, "length_bin": np.uint8, "location_mask": np.int64}
        for name, dtype in per_protein.items():
            array = getattr(self, name)
            if array.shape != (n_p,) or array.dtype != dtype:
                raise ValueError(f"{name} must be a ({n_p},) {np.dtype(dtype).name} array, not {array.shape} {array.dtype}")
        if self.go_terms.ndim != 2 or self.go_terms.shape[0] != n_p or self.go_terms.dtype != np.int16:
            raise ValueError(f"go_terms must be a ({n_p}, w) int16 array, not {self.go_terms.shape} {self.go_terms.dtype}")
        if list(self.proteins) != sorted(set(self.proteins)):
            raise ValueError("proteins must be sorted and unique")
        if n_p and (self.cluster.min() < 0 or self.cluster.max() >= self.n_clusters):
            raise ValueError("a cluster id lies outside [0, n_clusters)")
        if self.length_bin.size and int(self.length_bin.max()) >= LENGTH_BINS:
            raise ValueError(f"a length bin lies outside [0, {LENGTH_BINS})")
        if self.go_terms.size and (int(self.go_terms.max()) >= self.n_go_terms or int(self.go_terms.min()) < -1):
            raise ValueError("a GO term id lies outside [-1, n_go_terms)")
        for name in ("cluster_pair_keys", "exact_pair_keys"):
            keys = getattr(self, name)
            if keys.ndim != 1 or keys.dtype != np.int64 or keys.size == 0 or np.any(np.diff(keys) <= 0):
                raise ValueError(f"{name} must be a nonempty strictly increasing int64 vector")
        keep = self.keep_fraction
        if keep.shape != (STRATA,) or keep.dtype != np.float32 or not np.all((keep >= 0) & (keep <= 1)):
            raise ValueError(f"keep_fraction must be a ({STRATA},) float32 vector in [0, 1]")
        return self

    def with_keep_fraction(self, keep_fraction: np.ndarray) -> MaskTables:
        """The same tables with another keep fraction per stratum, `(n_s,)` of any float dtype."""
        # keep_fraction: (n_s,) any float dtype, converted to float32
        return self._replace(keep_fraction=keep_fraction.astype(np.float32)).checked()

    def save(self, path: Path) -> None:
        """Write the tables as one compressed `.npz`, which `load_tables` reads without pickling."""
        np.savez_compressed(
            path, schema=np.array(TABLE_SCHEMA), proteins=np.array(self.proteins, dtype=str), species=self.species, cluster=self.cluster,
            n_clusters=np.array(self.n_clusters), cluster_pair_keys=self.cluster_pair_keys, exact_pair_keys=self.exact_pair_keys,
            source_mask=self.source_mask, length_bin=self.length_bin, location_mask=self.location_mask, go_terms=self.go_terms,
            n_go_terms=np.array(self.n_go_terms), keep_fraction=self.keep_fraction,
        )


def load_tables(path: Path) -> MaskTables:
    """Read tables written by `MaskTables.save`, refusing a file of another schema or with an inconsistent array."""
    with np.load(path, allow_pickle=False) as stored:
        if str(stored["schema"]) != TABLE_SCHEMA:
            raise ValueError(f"{path} is not a {TABLE_SCHEMA} file")
        return MaskTables(
            tuple(str(item) for item in stored["proteins"]), stored["species"], stored["cluster"], int(stored["n_clusters"]),
            stored["cluster_pair_keys"], stored["exact_pair_keys"], stored["source_mask"], stored["length_bin"], stored["location_mask"],
            stored["go_terms"], int(stored["n_go_terms"]), stored["keep_fraction"],
        ).checked()


def check_rows(tables: MaskTables, rows: Sequence[Mapping[str, Any]]) -> None:
    """Refuse tables that were built from another release than the training rows they will mask.

    Each row carries `left`, `right`, `species_left`, `species_right`, `cluster_left` and `cluster_right`. Every row's two proteins
    must be in the tables under the species the row names, every row must be a known positive pair, and the rows' clusters must
    partition the proteins exactly as the tables' clusters do. Row clusters are labels of any type, so only the partition is compared.
    """
    index = {protein: position for position, protein in enumerate(tables.proteins)}
    try:
        left = np.array([index[row["left"]] for row in rows], dtype=np.int64)  # (n_r,)
        right = np.array([index[row["right"]] for row in rows], dtype=np.int64)  # (n_r,)
    except KeyError as error:
        raise ValueError(f"protein {error.args[0]} of a training row is not in the mask tables") from error
    species_left = np.array([row["species_left"] for row in rows], dtype=np.int64)  # (n_r,)
    species_right = np.array([row["species_right"] for row in rows], dtype=np.int64)  # (n_r,)
    if not (np.array_equal(tables.species[left], species_left) and np.array_equal(tables.species[right], species_right)):
        raise ValueError("the mask tables name a training protein by another species than the rows do")
    keys = np.minimum(left, right) * len(tables.proteins) + np.maximum(left, right)  # (n_r,)
    positions = np.searchsorted(tables.exact_pair_keys, keys)  # (n_r,)
    known = tables.exact_pair_keys[np.minimum(positions, tables.exact_pair_keys.size - 1)] == keys  # (n_r,)
    if not known.all():
        raise ValueError(f"{int((~known).sum()):,} training rows are not known positive pairs of the mask tables")
    proteins = np.concatenate((left, right))  # (2 n_r,)
    labels = np.array([str(row[side]) for side in ("cluster_left", "cluster_right") for row in rows])  # (2 n_r,)
    codes = np.unique(labels, return_inverse=True)[1].astype(np.int64)  # (2 n_r,)
    ids = tables.cluster[proteins]  # (2 n_r,)
    joint = np.unique(codes * (tables.n_clusters + 1) + ids).size  # distinct (label, cluster id) pairs
    if not (joint == np.unique(codes).size == np.unique(ids).size):
        raise ValueError("the mask tables cluster the training proteins differently than the rows do")


class Candidates(NamedTuple):
    """The cells of one batch after the taxonomy and homology checks, before the metadata check."""

    positive: Tensor  # (b, b) bool, a known positive pair in the same taxonomy pair; includes the diagonal
    taxonomic: Tensor  # (b, b) bool, an off-diagonal cell inside the row's taxonomy pair that is not a positive
    negative: Tensor  # (b, b) bool, of those, a cell that no known positive resembles


class Stages(NamedTuple):
    """Cell counts after each check of one batch, as 0-d int64 tensors on the device."""

    off_diagonal: Tensor  # () every cell but the diagonal
    taxonomic: Tensor  # () cells inside the row's taxonomy pair that are not positives
    homologous_kept: Tensor  # () of those, the cells no known positive resembles
    metadata_kept: Tensor  # () of those, the cells the metadata check keeps


class OffDiagonalMask:
    """Positive and eligible cells of a paired batch, computed on `device` from `MaskTables`.

    `batch(rows)` has the interface of the original Atlas `PositiveMask.batch`: it returns `(positive, eligible)`, both `(b, b)`
    bool, ready for `symmetric_multi_positive_loss`. Rows carry `left` and `right`, the protein ids of the tables. The metadata
    check draws from a generator seeded by `seed` and the step, so a resumed run draws the same cells; pass the training step.
    """

    def __init__(self, tables: MaskTables, device: torch.device | str = "cpu", *, seed: int = 0, go_dtype: torch.dtype | None = None) -> None:
        tables.checked()
        self.device = torch.device(device)
        self.seed = seed
        self.calls = 0
        self.index = {protein: position for position, protein in enumerate(tables.proteins)}
        self.n_proteins = len(tables.proteins)
        self.n_clusters = tables.n_clusters
        self.n_go_terms = tables.n_go_terms
        # A half-precision matrix product is exact for the integer term counts a protein has and much faster on a GPU.
        self.go_dtype = go_dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)

        def put(array: np.ndarray, dtype: torch.dtype | None = None) -> Tensor:
            # array: (...) any table array; the tensor has its shape
            return torch.tensor(array, dtype=dtype, device=self.device)  # (...)

        self.species = put(tables.species)  # (n_p,) int64
        self.cluster = put(tables.cluster)  # (n_p,) int64
        self.source_mask = put(tables.source_mask, torch.int64)  # (n_p,) int64, so that a bit test is one `&`
        self.length_bin = put(tables.length_bin, torch.int64)  # (n_p,) int64
        self.location_mask = put(tables.location_mask)  # (n_p,) int64
        self.go_terms = put(tables.go_terms)  # (n_p, w) int16, -1 padded
        self.cluster_pair_keys = put(tables.cluster_pair_keys)  # (n_k,) int64 sorted
        self.exact_pair_keys = put(tables.exact_pair_keys)  # (n_e,) int64 sorted
        self.keep_fraction = put(tables.keep_fraction)  # (n_s,) float32
        self.thins = bool(np.any(tables.keep_fraction < 1.0))
        self.edges = torch.tensor(JACCARD_EDGES, dtype=torch.float32, device=self.device)  # (4,)
        self.generator = torch.Generator(device=self.device)

    def indices(self, rows: Sequence[Mapping[str, Any]]) -> tuple[Tensor, Tensor]:
        """The table index of each row's left and right protein, `(b,)` int64 each, on the device."""
        try:
            left = [self.index[row["left"]] for row in rows]
            right = [self.index[row["right"]] for row in rows]
        except KeyError as error:
            raise ValueError(f"protein {error.args[0]} is not in the mask tables") from error
        return (torch.tensor(left, dtype=torch.int64, device=self.device),  # (b,)
                torch.tensor(right, dtype=torch.int64, device=self.device))  # (b,)

    @staticmethod
    def _contains(keys: Tensor, queries: Tensor) -> Tensor:
        """Whether each query is one of the sorted `keys`: `(n,)` and `(b, b)` in, `(b, b)` bool out."""
        # keys: (n,) sorted int64, nonempty; queries: (b, b) int64
        positions = torch.searchsorted(keys, queries)  # (b, b)
        return keys[positions.clamp(max=keys.numel() - 1)] == queries  # (b, b)

    @staticmethod
    def _pair_key(first: Tensor, second: Tensor, size: int) -> Tensor:
        """The unordered key of two ids below `size`: `(b, 1)` and `(1, b)` in, `(b, b)` int64 out, equal for either orientation."""
        # first: (b, 1) int64 ids; second: (1, b) int64 ids
        return torch.minimum(first, second) * size + torch.maximum(first, second)  # (b, b)

    def candidates(self, left: Tensor, right: Tensor) -> Candidates:
        """The taxonomy and homology checks of one batch, from the `(b,)` protein indices of its rows' left and right proteins."""
        # left, right: (b,) int64 table indices; cell (i, j) pairs left[i] with right[j]
        species_left, species_right = self.species[left], self.species[right]  # (b,), (b,)
        same_taxonomy = (species_left[:, None] == species_left[None, :]) & (species_right[:, None] == species_right[None, :])  # (b, b)
        cell_left, cell_right = left[:, None], right[None, :]  # (b, 1), (1, b)
        positive = self._contains(self.exact_pair_keys, self._pair_key(cell_left, cell_right, self.n_proteins)) & same_taxonomy  # (b, b)
        taxonomic = same_taxonomy & ~positive & (cell_left != cell_right)  # (b, b), a protein with itself is never a negative
        # The unordered cluster-pair key finds a known positive in either orientation.
        homologous = self._contains(
            self.cluster_pair_keys, self._pair_key(self.cluster[cell_left], self.cluster[cell_right], self.n_clusters))  # (b, b)
        return Candidates(positive, taxonomic, taxonomic & ~homologous)

    def _multi_hot(self, proteins: Tensor) -> Tensor:
        """Each protein's GO-BP terms as a 0/1 row: `(b,)` in, `(b, n_v + 1)` out; the last column absorbs the padding and stays 0."""
        # proteins: (b,) int64 table indices
        terms =self.go_terms[proteins].long()  # (b, w)
        columns = torch.where(terms >= 0, terms, torch.full_like(terms, self.n_go_terms))  # (b, w)
        matrix = torch.zeros(proteins.numel(), self.n_go_terms + 1, dtype=self.go_dtype, device=self.device)  # (b, n_v + 1)
        matrix.scatter_(1, columns, torch.ones_like(columns, dtype=self.go_dtype))  # (b, n_v + 1)
        matrix[:, self.n_go_terms] = 0
        return matrix  # (b, n_v + 1)

    def overlap_bins(self, left: Tensor, right: Tensor) -> Tensor:
        """The GO-BP Jaccard bin of every cell, `(b, b)` int64 in `[0, OVERLAP_BINS)`; proteins without terms share none."""
        # left, right: (b,) int64 table indices
        terms_left, terms_right = self._multi_hot(left), self._multi_hot(right)  # (b, n_v + 1) each
        intersection = (terms_left @ terms_right.T).float()  # (b, b)
        union = terms_left.sum(dim=1).float()[:, None] + terms_right.sum(dim=1).float()[None, :] - intersection  # (b, b)
        overlap = torch.where(union > 0, intersection / union.clamp(min=1.0), torch.zeros_like(union))  # (b, b)
        return (overlap[..., None] > self.edges).sum(dim=-1)  # (b, b)

    def strata(self, left: Tensor, right: Tensor) -> Tensor:
        """The stratum of every cell, `(b, b)` int64 in `[0, STRATA)`: overlap bin, shared location, shared source, length gap."""
        # left, right: (b,) int64 table indices
        shares_location = (self.location_mask[left][:, None] & self.location_mask[right][None, :]) != 0  # (b, b)
        shares_source = (self.source_mask[left][:, None] & self.source_mask[right][None, :]) != 0  # (b, b)
        length_gap = (self.length_bin[left][:, None] - self.length_bin[right][None, :]).abs().clamp(max=LENGTH_GAPS - 1)  # (b, b)
        overlap = self.overlap_bins(left, right)  # (b, b)
        return ((overlap * 2 + shares_location) * 2 + shares_source) * LENGTH_GAPS + length_gap  # (b, b)

    def paired_strata(self, left: Tensor, right: Tensor) -> Tensor:
        """Metadata codes for aligned pairs, without constructing all cross-pairs."""
        # left, right: (b,); multi-hot terms: (b, n_v + 1)
        terms_left, terms_right = self._multi_hot(left), self._multi_hot(right)  # (b, n_v + 1) each
        intersection = (terms_left * terms_right).sum(dim=1).float()  # (b,)
        union = terms_left.sum(dim=1).float() + terms_right.sum(dim=1).float() - intersection  # (b,)
        jaccard = torch.where(union > 0, intersection / union.clamp_min(1), torch.zeros_like(union))  # (b,)
        overlap = (jaccard[:, None] > self.edges).sum(dim=1)  # (b,)
        location = (self.location_mask[left] & self.location_mask[right]) != 0  # (b,)
        source = (self.source_mask[left] & self.source_mask[right]) != 0  # (b,)
        gap = (self.length_bin[left] - self.length_bin[right]).abs().clamp(max=LENGTH_GAPS - 1)  # (b,)
        return ((overlap * 2 + location) * 2 + source) * LENGTH_GAPS + gap  # (b,)

    def _kept(self, negative: Tensor, left: Tensor, right: Tensor, step: int | None) -> Tensor:
        """The candidates the metadata check keeps, `(b, b)` bool: each is kept with its stratum's keep fraction."""
        # negative: (b, b) bool, True = candidate; left, right: (b,) int64 table indices
        if not self.thins:
            return negative  # (b, b)
        self.generator.manual_seed(self.seed * 1_000_003 + (self.calls if step is None else step))
        self.calls += 1
        draws = torch.rand(tuple(negative.shape), generator=self.generator, device=self.device)  # (b, b)
        return negative & (draws < self.keep_fraction[self.strata(left, right)])  # (b, b)

    def masks(self, left: Tensor, right: Tensor, step: int | None = None) -> tuple[Tensor, Tensor]:
        """`(positive, eligible)`, both `(b, b)` bool, from the `(b,)` protein indices of a batch's rows."""
        # left, right: (b,) int64 table indices
        positive, _, negative = self.candidates(left, right)  # (b, b) each
        return positive, positive | self._kept(negative, left, right, step)  # (b, b) each

    def stages(self, left: Tensor, right: Tensor, step: int | None = None) -> Stages:
        """How many cells each check leaves, for the same `step` that `masks` was given."""
        # left, right: (b,) int64 table indices
        _, taxonomic, negative = self.candidates(left, right)  # (b, b) each
        rows = left.numel()
        return Stages(torch.tensor(rows * (rows - 1), device=self.device), taxonomic.sum(), negative.sum(),
                      self._kept(negative, left, right, step).sum())

    def batch(self, rows: Sequence[Mapping[str, Any]], step: int | None = None) -> tuple[Tensor, Tensor]:
        """`(positive, eligible)`, both `(b, b)` bool on the device, for rows that each carry `left` and `right` protein ids."""
        left, right = self.indices(rows)  # (b,), (b,)
        return self.masks(left, right, step)  # (b, b) each


def open_mask(path: Path, *, expected_sha256: str, rows: Sequence[Mapping[str, Any]], device: torch.device | str, seed: int = 0) -> OffDiagonalMask:
    """The mask of a tables file held by its independently recorded SHA-256 and checked against the rows it will serve."""
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{path} differs from its pinned SHA-256")
    tables = load_tables(path)
    check_rows(tables, rows)
    return OffDiagonalMask(tables, device, seed=seed)
