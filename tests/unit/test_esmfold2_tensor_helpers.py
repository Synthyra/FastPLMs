"""The ESMFold2 tensor, sequence, and annotation helpers, and the dataclass that keeps residue fields aligned.

The helpers carry no model state, so each test hands one a small input whose answer is clear by hand.

Shapes: `b` batch rows, `l` residues, `n` points, `k` neighbors per residue.
"""

import warnings
import numpy as np
import pytest
import torch

from dataclasses import dataclass, field

from fastplms.models.esmfold2 import esmfold2_misc as misc
from fastplms.models.esmfold2.esmfold2_sequential_dataclass import SequentialDataclass
from fastplms.models.esmfold2.esmfold2_utils_types import FunctionAnnotation


@dataclass(frozen=True)
class Chain(SequentialDataclass):
    """Residues with a letter and a position each, and integer tracks aligned by track. Each joins with a separator."""

    letters: str = field(metadata={"sequence": True, "join_token": "|"})
    positions: np.ndarray = field(metadata={"sequence": True, "join_token": -1.0})  # (l,)
    tracks: list[list[int]] | None = field(
        default=None, metadata={"sequence": True, "sequence_dim": 1, "join_token": 0}
    )
    name: str = "chain"

    def __len__(self) -> int:
        return len(self.letters)


@dataclass(frozen=True)
class Misdeclared(SequentialDataclass):
    values: list[int] = field(metadata={"sequence": True, "sequence_dim": 2})

    def __len__(self) -> int:
        return len(self.values)


class Joinable:
    """A record that knows how to concatenate a list of its own kind."""

    def __init__(self, parts: list[int]) -> None:
        self.parts = parts

    @classmethod
    def concat(cls, objs: list["Joinable"]) -> "Joinable":
        return cls([part for item in objs for part in item.parts])


def chain(letters: str, offset: float = 0.0, tracks: list[list[int]] | None = None) -> Chain:
    positions = np.arange(len(letters), dtype=np.float32) + offset  # (l,)
    return Chain(letters=letters, positions=positions, tracks=tracks)


def test_the_fp32_context_matches_the_device_and_rejects_unknown_devices():
    with misc.fp32_autocast_context("cpu"):
        assert not torch.is_autocast_enabled("cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # without CUDA, torch warns that it disables the autocast it was asked for
        assert misc.fp32_autocast_context("cuda") is not None
    with misc.fp32_autocast_context("mps"):
        pass
    with pytest.raises(ValueError, match="Unsupported device type"):
        misc.fp32_autocast_context("tpu")


def test_an_optional_value_becomes_a_tensor_and_none_stays_none():
    stacked = misc.maybe_tensor([torch.zeros(2), torch.ones(2)])  # (2, 2)
    with_gaps = misc.maybe_tensor([[1.0, None]], convert_none_to_nan=True)  # (1, 2)
    existing = torch.arange(3)  # (3,)

    assert misc.maybe_tensor(None) is None
    assert misc.maybe_tensor(existing) is existing
    assert stacked.shape == (2, 2)
    assert misc.maybe_tensor([[1, 2], [3, 4]]).tolist() == [[1, 2], [3, 4]]
    assert with_gaps[0, 0] == 1.0 and torch.isnan(with_gaps[0, 1])


def test_an_optional_array_becomes_nested_lists_with_nan_as_none():
    tensor = torch.tensor([[1.0, float("nan")]])  # (1, 2)
    array = np.array([[1.0, np.nan]])  # (1, 2)

    assert misc.maybe_list(None) is None
    assert misc.maybe_list(tensor)[0][0] == 1.0
    assert misc.maybe_list(tensor, convert_nan_to_none=True) == [[1.0, None]]
    assert misc.maybe_list(array, convert_nan_to_none=True) == [[1.0, None]]
    with pytest.raises(TypeError, match="torch.tensor or np.ndarray"):
        misc.maybe_list([1.0], convert_nan_to_none=True)


def test_infinite_values_become_the_api_sentinel():
    assert misc.replace_inf(None) is None
    assert misc.replace_inf([[1.0, float("inf")], [float("-inf"), 2.0]]) == [[1.0, 1000.0], [1000.0, 2.0]]


def test_python_sequences_slice_like_arrays_and_arrays_slice_as_they_are():
    letters = "ACDEF"
    items = [10, 11, 12, 13]

    assert misc.slice_any_object(letters, 2) == "D"
    assert misc.slice_any_object(letters, slice(1, 3)) == "CD"
    assert misc.slice_any_object(letters, [0, 4]) == "AF"
    assert misc.slice_any_object(letters, np.array([True, False, True, False, False])) == "AD"
    assert misc.slice_any_object(items, np.array([False, True, False, True])) == [11, 13]
    assert misc.slice_any_object(tuple(items), slice(None, None, 2)) == (10, 12)
    assert misc.slice_any_object(np.arange(5), slice(1, 3)).tolist() == [1, 2]
    assert misc.slice_any_object(torch.arange(5), [0, 4]).tolist() == [0, 4]
    assert misc.slice_any_object(chain("ACD"), slice(0, 2)).letters == "AC"


def test_lists_join_with_an_optional_separator():
    assert misc.join_lists([]) == []
    assert misc.join_lists([[1, 2]]) == [1, 2]
    assert misc.join_lists([[1], [2], [3]], separator=[0, 0]) == [1, 0, 0, 2, 0, 0, 3]
    assert list(misc.iterate_with_intermediate("abc", "-")) == ["a", "-", "b", "-", "c"]


def test_collections_of_every_supported_kind_concatenate():
    joined = misc.concat_objects([Joinable([1]), Joinable([2, 3])])

    assert joined.parts == [1, 2, 3]
    assert misc.concat_objects(["AC", "DE"], "|") == "AC|DE"
    assert misc.concat_objects([[1], [2]], 0) == [1, 0, 2]
    assert misc.concat_objects([[1], [2]]) == [1, 2]
    assert misc.concat_objects([np.array([1, 2]), np.array([3])], 9).tolist() == [1, 2, 9, 3]
    assert misc.concat_objects([np.array([1, 2]), np.array([3])]).tolist() == [1, 2, 3]
    assert misc.concat_objects([torch.tensor([1]), torch.tensor([2])], 7).tolist() == [1, 7, 2]
    assert misc.concat_objects([torch.tensor([1]), torch.tensor([2])]).tolist() == [1, 2]


def test_concatenating_nothing_or_the_wrong_kind_is_an_error():
    with pytest.raises(ValueError, match="at least one value"):
        misc.concat_objects([])
    with pytest.raises(TypeError, match="separator must be a string"):
        misc.concat_objects(["AC", "DE"])
    with pytest.raises(TypeError):
        misc.concat_objects([3.5, 4.5])


def test_radial_basis_features_peak_at_the_center_a_value_sits_on():
    values = torch.tensor([0.0, 1.0, 2.0])  # (n,)

    features = misc.rbf(values, v_min=0.0, v_max=2.0, n_bins=3)  # (n, n_bins)

    assert features.shape == (3, 3)
    torch.testing.assert_close(features.diagonal(), torch.ones(3))
    assert (features <= 1.0).all() and (features > 0.0).all()


def test_batched_gather_picks_the_same_indices_in_every_batch_row():
    table = torch.arange(24).reshape(2, 3, 4)  # (b, 3, 4)
    indices = torch.tensor([[2, 0], [1, 1]])  # (b, 2)

    gathered = misc.batched_gather(table, indices, dim=1, no_batch_dims=1)  # (b, 2, 4)

    assert torch.equal(gathered[0, 0], table[0, 2])
    assert torch.equal(gathered[1, 1], table[1, 1])
    assert gathered.shape == (2, 2, 4)


def test_node_gather_collects_the_features_of_each_residues_neighbors():
    features = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)  # (b, l, d)
    neighbors = torch.tensor([[[1, 2], [0, 3], [3, 0], [2, 1]]])  # (b, l, k)

    gathered = misc.node_gather(features, neighbors)  # (b, l, k, d)

    assert gathered.shape == (1, 4, 2, 3)
    assert torch.equal(gathered[0, 0, 0], features[0, 1])
    assert torch.equal(gathered[0, 3, 1], features[0, 1])


def test_nearest_neighbors_are_the_closest_residues_and_missing_geometry_sorts_last():
    positions = torch.tensor([[[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [10.0, 0, 0]]])  # (b, l, 3)
    everything_present = torch.ones(1, 5, dtype=torch.bool)  # (b, l)
    nothing_padded = torch.zeros(1, 5, dtype=torch.bool)  # (b, l)
    one_chain = torch.zeros(1, 5, dtype=torch.long)  # (b, l)
    unresolved_third = everything_present.clone()
    unresolved_third[0, 2] = False
    padded_last = nothing_padded.clone()
    padded_last[0, 4] = True

    edges, valid = misc.knn_graph(positions, everything_present, nothing_padded, one_chain, no_knn=3)  # (b, l, k) each
    all_edges, _ = misc.knn_graph(positions, unresolved_third, nothing_padded, one_chain, no_knn=9)  # (b, l, l)
    _, padded_valid = misc.knn_graph(positions, everything_present, padded_last, one_chain, no_knn=5)  # (b, l, l)

    assert edges[0, 0].tolist() == [0, 1, 2]
    assert edges[0, 4].tolist() == [4, 3, 2]
    assert valid.all()
    assert all_edges[0, 0].tolist()[-1] == 2  # the residue without coordinates is the last neighbor
    assert not padded_valid[0, 0, 4] and padded_valid[0, 0, :4].all()


def test_neighbors_of_residues_in_different_chains_are_excluded_and_absurd_distances_are_refused():
    positions = torch.tensor([[[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]])  # (b, l, 3)
    present = torch.ones(1, 3, dtype=torch.bool)  # (b, l)
    padded = torch.zeros(1, 3, dtype=torch.bool)  # (b, l)
    two_chains = torch.tensor([[0, 0, 1]])  # (b, l)
    far_apart = torch.tensor([[[0.0, 0, 0], [2.0e6, 0, 0], [1.0, 0, 0]]])  # (b, l, 3)

    _, valid = misc.knn_graph(positions, present, padded, two_chains, no_knn=3)  # (b, l, k)

    assert valid[0, 0].tolist() == [True, True, False]
    with pytest.raises(ValueError, match="exceed max supported distance"):
        misc.knn_graph(far_apart, present, padded, torch.zeros(1, 3, dtype=torch.long), no_knn=2)


def test_tensors_of_different_lengths_stack_with_padding():
    short = torch.ones(2, 3)  # (2, 3)
    long = torch.full((4, 2), 2.0)  # (4, 2)

    stacked = misc.stack_variable_length_tensors([short, long], constant_value=-1)  # (2, 4, 3)
    retyped = misc.stack_variable_length_tensors([torch.ones(1), torch.ones(3)], dtype=torch.float64)  # (2, 3)

    assert stacked.shape == (2, 4, 3)
    assert torch.equal(stacked[0, :2, :3], short) and (stacked[0, 2:] == -1).all()
    assert torch.equal(stacked[1, :4, :2], long) and (stacked[1, :, 2] == -1).all()
    assert retyped.dtype == torch.float64 and retyped.tolist() == [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]


def test_packing_and_unpacking_sequences_by_id_are_inverse_operations():
    sequence_id = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 0, 0]])  # (b, packed_length)
    rows = torch.tensor(
        [
            [1.0, 2.0, 0.0, 0.0, 0.0],
            [3.0, 4.0, 5.0, 0.0, 0.0],
            [6.0, 7.0, 8.0, 9.0, 10.0],
        ]
    )  # (n_unpacked_sequences, max_length)

    packed = misc.binpack(rows, sequence_id, pad_value=0)  # (b, packed_length)
    unpacked = misc.unbinpack(packed, sequence_id, pad_value=0)  # (n_unpacked_sequences, max_length)

    assert packed.tolist() == [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]
    assert torch.equal(unpacked, rows)
    assert misc.binpack(rows, None, 0) is rows and misc.unbinpack(rows, None, 0) is rows


def test_overlapping_and_adjacent_ranges_merge_by_the_allowed_gap():
    ranges = [range(10, 12), range(0, 3), range(2, 5), range(6, 8)]

    assert misc.merge_ranges(ranges) == [range(0, 5), range(6, 8), range(10, 12)]
    assert misc.merge_ranges(ranges, merge_gap_max=1) == [range(0, 8), range(10, 12)]
    assert misc.merge_ranges(ranges, merge_gap_max=2) == [range(0, 12)]
    with pytest.raises(TypeError, match="integer or None"):
        misc.merge_ranges(ranges, merge_gap_max=1.5)
    with pytest.raises(ValueError, match="non-negative"):
        misc.merge_ranges(ranges, merge_gap_max=-1)


def test_annotations_merge_within_their_own_label_only():
    annotations = [
        FunctionAnnotation("helix", 1, 4),
        FunctionAnnotation("helix", 3, 8),
        FunctionAnnotation("sheet", 5, 6),
        FunctionAnnotation("helix", 20, 21),
    ]

    merged = misc.merge_annotations(annotations)

    assert [(item.label, item.start, item.end) for item in merged] == [("helix", 1, 8), ("helix", 20, 21), ("sheet", 5, 6)]
    assert [item.to_tuple() for item in merged][0] == ("helix", 1, 8)
    assert [len(item) for item in merged] == [8, 2, 2]


def test_chain_breaks_split_a_sequence_into_half_open_chains():
    boundaries = misc.get_chainbreak_boundaries_from_sequence("AAA|BB|C" + "C")  # (n_chains, 2)

    assert boundaries.tolist() == [[0, 3], [4, 6], [7, 9]]
    assert misc.get_chainbreak_boundaries_from_sequence("AAA").tolist() == [[0, 3]]
    with pytest.warns(UserWarning, match="penultimate"):
        misc.get_chainbreak_boundaries_from_sequence("AA|B")
    with pytest.raises(ValueError, match="end of sequence"):
        misc.get_chainbreak_boundaries_from_sequence("AA|")


def test_aligned_fields_check_their_lengths_and_choose_the_axis_that_holds_them():
    with pytest.raises(ValueError, match="Mismatch in sequence length for field: positions"):
        Chain(letters="ACD", positions=np.zeros(2))
    with pytest.raises(ValueError, match="field: tracks"):
        Chain(letters="AC", positions=np.zeros(2), tracks=[[1, 2], [3]])
    with pytest.raises(NotImplementedError, match="zero and one"):
        Misdeclared(values=[1, 2])


def test_indexing_an_aligned_record_slices_every_aligned_field_and_keeps_the_rest():
    record = chain("ACDEF", tracks=[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

    second = record[1]
    window = record[1:3]
    picked = record[[0, 4]]

    assert (second.letters, window.letters, picked.letters) == ("C", "CD", "AF")
    assert window.positions.tolist() == [1.0, 2.0] and window.tracks == [[2, 3], [7, 8]]
    assert picked.tracks == [[1, 5], [6, 10]] and picked.name == "chain"
    assert chain("AC")[0:2].tracks is None


def test_aligned_records_concatenate_with_each_fields_join_token():
    first = chain("AC", tracks=[[1, 2]])
    second = chain("DEF", offset=5.0, tracks=[[3, 4, 5]])

    joined = Chain.concat([first, second], name="joined")

    assert joined.letters == "AC|DEF"
    assert joined.positions.tolist() == [0.0, 1.0, -1.0, 5.0, 6.0, 7.0]
    assert joined.tracks == [[1, 2, 0, 3, 4, 5]]
    assert joined.name == "joined" and Chain.concat([first]).name == "chain"
    with pytest.raises(ValueError, match="at least one item"):
        Chain.concat([])


def test_an_annotation_counts_its_inclusive_residues():
    annotation = FunctionAnnotation(label="domain", start=5, end=9)

    assert annotation.to_tuple() == ("domain", 5, 9)
    assert len(annotation) == 5
