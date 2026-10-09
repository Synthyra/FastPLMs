"""Multiple-sequence alignments for ESMFold2: A3M and Stockholm input, selection, serialization, and taxonomy pairing.

Each alignment is a few rows of a four-column query, so a selected row, a decoded insertion count, or a paired row can
be checked against the text that produced it.

Shapes: `m` alignment rows, `l` alignment columns, `t` tokens of a multichain input.
"""

import io
import numpy as np
import pytest

from fastplms.models.esmfold2 import esmfold2_msa as msa_module
from fastplms.models.esmfold2 import esmfold2_paired_msa as paired
from fastplms.models.esmfold2.esmfold2_constants import (
    MSA_GAP_TOKEN_ID,
    PROTEIN_RESIDUE_TO_RES_TYPE,
    PROTEIN_UNK_RES_TYPE,
)
from fastplms.models.esmfold2.esmfold2_msa import FastMSA, MSA, a3m_deletion_counts, is_a3m_insertion
from fastplms.models.esmfold2.esmfold2_parsing import FastaEntry


A3M = ">query\nACDE\n>hit one key=100\nAC-E\n>hit two key=200\nAcxCDE\n"
STOCKHOLM = "# STOCKHOLM 1.0\nquery ACD-E\nother AC-DE\n//\n"
ALANINE = PROTEIN_RESIDUE_TO_RES_TYPE["ALA"]
CYSTEINE = PROTEIN_RESIDUE_TO_RES_TYPE["CYS"]
ASPARTATE = PROTEIN_RESIDUE_TO_RES_TYPE["ASP"]
GLUTAMATE = PROTEIN_RESIDUE_TO_RES_TYPE["GLU"]


def alignment() -> MSA:
    return MSA.from_a3m(io.StringIO(A3M))


def test_insertion_markers_are_lowercase_letters_and_dots():
    assert is_a3m_insertion(".") and is_a3m_insertion("c")
    assert not is_a3m_insertion("C") and not is_a3m_insertion("-")
    assert msa_module.remove_insertions_from_sequence("AcxC.DE") == "ACDE"
    assert a3m_deletion_counts("AcxCDE").tolist() == [0, 2, 0, 0]


def test_an_a3m_alignment_reads_from_a_stream_or_a_file_and_counts_its_insertions(tmp_path):
    path = tmp_path / "alignment.a3m"
    path.write_text(A3M, encoding="utf-8")

    from_stream = alignment()
    from_path = MSA.from_a3m(path)
    unchanged = MSA.from_a3m(io.StringIO(">q\nACDE\n>r\nAC-E\n"), remove_insertions=False)
    first_two = MSA.from_a3m(io.StringIO(A3M), max_sequences=2)

    assert from_stream.sequences == ["ACDE", "AC-E", "ACDE"]
    assert from_stream.headers == ["query", "hit one key=100", "hit two key=200"]
    assert from_stream.deletions.tolist() == [[0, 0, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0]]
    assert from_path.entries == from_stream.entries
    assert unchanged.sequences == ["ACDE", "AC-E"] and unchanged.deletions is None
    assert first_two.depth == 2
    with pytest.raises(ValueError, match="Sequence length mismatch"):
        MSA.from_a3m(io.StringIO(A3M), remove_insertions=False)  # the raw row with insertions is wider than the query


def test_an_a3m_alignment_with_rows_of_different_widths_is_refused():
    with pytest.raises(ValueError, match="Sequence length mismatch"):
        MSA.from_a3m(io.StringIO(">q\nACDE\n>r\nACD\n"))


def test_an_alignment_validates_its_rows_and_deletion_matrix():
    with pytest.raises(TypeError, match="list of FastaEntry"):
        MSA(("ACDE",))
    with pytest.raises(ValueError, match="at least one"):
        MSA([])
    with pytest.raises(TypeError, match="must be a FastaEntry"):
        MSA([FastaEntry("q", "ACDE"), ("r", "ACDE")])
    with pytest.raises(ValueError, match="non-empty"):
        MSA([FastaEntry("q", "")])
    with pytest.raises(ValueError, match="row length mismatch"):
        MSA([FastaEntry("q", "ACDE"), FastaEntry("r", "AC")])
    with pytest.raises(TypeError, match="NumPy array"):
        MSA([FastaEntry("q", "ACDE")], deletions=[[0, 0, 0, 0]])
    with pytest.raises(ValueError, match="deletion matrix"):
        MSA([FastaEntry("q", "ACDE")], deletions=np.zeros((2, 4)))


def test_an_alignment_reports_its_shape_query_identity_and_text_form():
    msa = alignment()

    assert (msa.depth, msa.seqlen, len(msa), msa.query) == (3, 4, 4, "ACDE")
    assert msa.array.shape == (3, 4) and msa.array.dtype == np.dtype("|S1")
    assert msa.seqid.tolist() == pytest.approx([1.0, 0.75, 1.0])
    assert repr(msa) == "MSA(query: Depth=3, Length=4)"


def test_a_stockholm_alignment_drops_the_columns_the_query_leaves_empty():
    msa = MSA.from_stockholm(io.StringIO(STOCKHOLM))
    kept = MSA.from_stockholm(io.StringIO(STOCKHOLM), remove_insertions=False)
    first = MSA.from_stockholm(io.StringIO(STOCKHOLM), max_sequences=1)

    assert msa.sequences == ["ACDE", "AC-E"]
    assert kept.sequences == ["ACD-E", "AC-DE"]
    assert msa.headers[0].startswith("query") and first.depth == 1
    with pytest.raises(ValueError):
        MSA.from_stockholm(io.StringIO("# STOCKHOLM 1.0\nquery ACDE\nother ACDEF\n//\n"))  # Biopython refuses ragged rows


def test_sequences_alone_make_an_alignment_with_or_without_their_insertions():
    kept = MSA.from_sequences(["ACDE", "AcCDE"])
    stripped = MSA.from_sequences(["ACDE", "AcCDE"], remove_insertions=True)

    assert kept.sequences == ["ACDE", "AcCDE"] and kept.headers == ["", ""]
    assert stripped.sequences == ["ACDE", "ACDE"]


def test_an_alignment_survives_every_serialization_it_offers():
    msa = alignment()

    from_bytes = MSA.from_bytes(msa.to_bytes())
    from_sequence_bytes = MSA.from_sequence_bytes(msa.to_sequence_bytes())
    exact_state = msa.state_dict()
    json_state = msa.state_dict(json_serializable=True)
    from_state = MSA.from_state_dict(json_state)
    without_deletions = MSA.from_sequences(["ACDE"])

    assert from_bytes.entries == msa.entries
    assert from_sequence_bytes.sequences == msa.sequences and from_sequence_bytes.headers == ["", "", ""]
    assert isinstance(exact_state["deletions"], np.ndarray) and isinstance(json_state["deletions"], list)
    assert from_state.sequences == msa.sequences and np.array_equal(from_state.deletions, msa.deletions)
    assert "deletions" not in without_deletions.state_dict()
    assert MSA.from_state_dict(without_deletions.state_dict()).deletions is None


def test_an_alignment_writes_a3m_text_and_converts_to_the_array_form():
    msa = alignment()
    stream = io.StringIO()

    msa.to_a3m(stream)
    fast = msa.to_fast_msa()

    assert stream.getvalue() == ">query\nACDE\n>hit one key=100\nAC-E\n>hit two key=200\nACDE"
    assert isinstance(fast, FastMSA) and fast.depth == 3 and fast.headers == msa.headers
    assert msa_module._parse_full_payload(msa.to_bytes())[0].shape == (3, 4)


def test_a_serialized_alignment_of_another_version_is_refused():
    payload = bytearray(alignment().to_bytes())
    payload[0] = 9

    with pytest.raises(ValueError, match="Unsupported version: 9"):
        MSA.from_bytes(bytes(payload))
    headerless = msa_module._full_payload(np.array([[b"A", b"C"]], dtype="|S1"), [])
    assert MSA.from_bytes(headerless).headers == [""]


def test_rows_and_columns_are_selected_with_their_deletions():
    msa = alignment()

    rows = msa.select_sequences([0, 2])
    columns = msa.select_positions([1, 3])
    one_column = msa[1]
    window = msa[1:3]
    picked = msa[[0, 3]]

    assert rows.sequences == ["ACDE", "ACDE"] and rows.deletions.tolist() == [[0, 0, 0, 0], [0, 2, 0, 0]]
    assert columns.sequences == ["CE", "CE", "CE"] and columns.deletions.tolist() == [[0, 0], [0, 0], [2, 0]]
    assert one_column.sequences == ["C", "C", "C"]
    assert window.sequences == ["CD", "C-", "CD"]
    assert picked.sequences == ["AE", "AE", "AE"]
    assert msa._aligned_deletions() is msa.deletions
    assert MSA.from_sequences(["ACDE"])._select_deletion_columns([0]) is None


def test_the_greedy_selection_keeps_the_query_and_then_the_farthest_or_nearest_rows():
    msa = MSA.from_sequences(["AAAA", "AAAC", "CCCC", "AACC"])

    farthest = msa.greedy_select(2)
    nearest = msa.greedy_select(2, mode="min")

    assert farthest.sequences == ["AAAA", "CCCC"]
    assert nearest.sequences == ["AAAA", "AAAC"]
    assert msa.greedy_select(10) is msa
    with pytest.raises(ValueError, match="Unsupported MSA selection mode"):
        msa.greedy_select(2, mode="median")


def test_random_selection_keeps_the_query_and_returns_the_requested_depth():
    msa = MSA.from_sequences(["AAAA", "AAAC", "CCCC", "AACC", "ACAC"])
    np.random.seed(0)

    chosen = msa.select_random_sequences(3)

    assert chosen.depth == 3 and chosen.sequences[0] == "AAAA"
    assert msa.select_random_sequences(5) is msa


def test_the_hhfilter_selection_forwards_its_thresholds_and_keeps_the_rows_the_filter_names(monkeypatch):
    calls = []

    def fake_hhfilter(sequences, **options):
        calls.append((list(sequences), options))
        return [0, 2, 3]

    monkeypatch.setattr(msa_module, "hhfilter", fake_hhfilter)
    msa = MSA.from_sequences(["AAAA", "AAAC", "CCCC", "AACC"])

    filtered = msa.hhfilter(seqid=80, diff=2, cov=10, qid=5, qsc=-5.0, binary="/opt/hh/hhfilter")
    diverse = msa.select_diverse_sequences(2)
    all_kept = msa.select_diverse_sequences(4)

    assert filtered.sequences == ["AAAA", "CCCC", "AACC"]
    assert calls[0][1] == {"seqid": 80, "diff": 2, "cov": 10, "qid": 5, "qsc": -5.0, "binary": "/opt/hh/hhfilter"}
    assert diverse.depth == 2 and diverse.sequences[0] == "AAAA"
    assert calls[1][1]["diff"] == 2
    assert all_kept is msa


def test_an_alignment_pads_with_gap_rows_and_stacks_without_repeating_the_query():
    msa = alignment()
    other = MSA.from_a3m(io.StringIO(">query\nACDE\n>more\nAAAA\n"))

    padded = msa.pad_to_depth(5)
    stacked = MSA.stack([msa, other])
    stacked_with_queries = MSA.stack([msa, other], remove_query_from_later_msas=False)

    assert padded.sequences[3:] == ["----", "----"] and padded.deletions.shape == (5, 4)
    assert not padded.deletions[3:].any()
    assert msa.pad_to_depth(3) is msa
    assert stacked.sequences == ["ACDE", "AC-E", "ACDE", "AAAA"] and stacked.deletions.shape == (4, 4)
    assert stacked_with_queries.depth == 5
    with pytest.raises(ValueError, match="Cannot pad to depth 2"):
        msa.pad_to_depth(2)


def test_alignments_of_the_same_depth_concatenate_by_column_with_or_without_a_separator():
    first = alignment()
    second = MSA.from_a3m(io.StringIO(">q2\nFG\n>b\nHI\n>c\nKL\n"))
    shallow = MSA.from_a3m(io.StringIO(">q3\nMN\n"))

    joined = MSA.concat([first, second])
    fused = MSA.concat([first, second], join_token=None)
    padded = MSA.concat([first, shallow], allow_depth_mismatch=True)

    assert joined.sequences == ["ACDE|FG", "AC-E|HI", "ACDE|KL"] and joined.deletions is None
    assert fused.sequences == ["ACDEFG", "AC-EHI", "ACDEKL"] and fused.deletions.shape == (3, 6)
    assert padded.sequences[2].endswith("|--") and padded.headers[0] == "query|q3"
    with pytest.raises(ValueError, match="Depth mismatch"):
        MSA.concat([first, shallow])
    with pytest.raises(ValueError, match="empty list"):
        MSA.concat([])


def test_concatenated_alignments_keep_their_deletions_when_nothing_separates_them():
    first = MSA.from_a3m(io.StringIO(">q\nAC\n>r\nAcC\n"))
    second = MSA.from_a3m(io.StringIO(">q\nDE\n>r\nDdE\n"))

    fused = MSA.concat([first, second], join_token=None)

    assert fused.deletions.tolist() == [[0, 0, 0, 0], [0, 1, 0, 1]]


def test_a_fast_alignment_checks_its_array_and_headers():
    array = np.array([[b"A", b"C"], [b"A", b"-"]], dtype="|S1")  # (m, l)

    assert FastMSA(array).headers is None
    with pytest.raises(TypeError, match="NumPy array"):
        FastMSA([["A", "C"]])
    with pytest.raises(ValueError, match="non-empty shape"):
        FastMSA(np.zeros((0, 2), dtype="|S1"))
    with pytest.raises(ValueError, match="Number of headers"):
        FastMSA(array, ["only one"])


def test_a_fast_alignment_serializes_selects_pads_and_converts_back():
    array = np.array([[b"A", b"C", b"D"], [b"A", b"-", b"D"], [b"E", b"C", b"D"]], dtype="|S1")  # (m, l)
    fast = FastMSA(array, ["query", "second", "third"])

    from_bytes = FastMSA.from_bytes(msa_module._full_payload(array, fast.headers))
    from_sequence_bytes = FastMSA.from_sequence_bytes(msa_module._sequence_payload(array))
    columns = fast[1]
    window = fast[1:3]
    rows = fast.select_sequences([0, 2])
    np.random.seed(0)
    sampled = fast.select_random_sequences(2)
    padded = fast.pad_to_depth(5)
    as_text = fast.to_msa()

    assert (fast.depth, fast.seqlen, len(fast)) == (3, 3, 3)
    assert from_bytes.headers == fast.headers and from_sequence_bytes.headers is None
    assert columns.array.tolist() == [[b"C"], [b"-"], [b"C"]] and window.array.shape == (3, 2)
    assert rows.headers == ["query", "third"] and rows.array.shape == (2, 3)
    assert sampled.depth == 2 and sampled.headers[0] == "query" and fast.select_random_sequences(3) is fast
    assert padded.depth == 5 and padded.headers[3:] == ["", ""] and (padded.array[3:] == b"-").all()
    assert fast.pad_to_depth(3) is fast
    assert as_text.sequences == ["ACD", "A-D", "ECD"] and as_text.headers == ["query", "second", "third"]
    assert FastMSA(array).to_msa().headers == ["seq0", "seq1", "seq2"]
    with pytest.raises(ValueError, match="Cannot pad to depth 2"):
        fast.pad_to_depth(2)


def test_fast_alignments_pad_unsigned_byte_arrays_with_the_gap_code():
    array = np.array([[ord("A"), ord("C")]], dtype=np.uint8)  # (m, l)

    padded = FastMSA(array).pad_to_depth(2)

    assert padded.array[1].tolist() == [ord("-"), ord("-")]


def test_fast_alignments_concatenate_by_column_and_stack_by_row():
    first = FastMSA(np.array([[b"A", b"C"], [b"A", b"-"]], dtype="|S1"), ["q1", "r1"])
    second = FastMSA(np.array([[b"D"], [b"E"]], dtype="|S1"))
    shallow = FastMSA(np.array([[b"F"]], dtype="|S1"), ["q3"])

    joined = FastMSA.concat([first, second])
    padded = FastMSA.concat([first, shallow], allow_depth_mismatch=True)
    stacked = FastMSA.stack([first, first])
    stacked_all = FastMSA.stack([first, first], remove_query_from_later_msas=False)
    unnamed = FastMSA.stack([second, second])

    assert joined.array.tolist() == [[b"A", b"C", b"D"], [b"A", b"-", b"E"]] and joined.headers == ["q1|", "r1|"]
    assert padded.array.shape == (2, 3) and padded.array[1, 2] == b"-"
    assert stacked.depth == 3 and stacked.headers == ["q1", "r1", "r1"] and stacked_all.depth == 4
    assert unnamed.headers is None
    with pytest.raises(NotImplementedError, match="join_token"):
        FastMSA.concat([first, second], join_token="|")
    with pytest.raises(ValueError, match="Depth mismatch"):
        FastMSA.concat([first, shallow])
    with pytest.raises(ValueError, match="empty list"):
        FastMSA.concat([])
    with pytest.raises(ValueError, match="empty list"):
        FastMSA.stack([])


def test_the_msa_vocabulary_maps_residues_a_gap_and_the_unknown_letter():
    vocabulary = paired.protein_letter_to_res_type()

    assert vocabulary["A"] == ALANINE and vocabulary["C"] == CYSTEINE
    assert vocabulary["-"] == MSA_GAP_TOKEN_ID and vocabulary["X"] == PROTEIN_UNK_RES_TYPE


def test_a_taxonomy_key_comes_from_the_header_and_defaults_to_minus_one():
    assert paired._taxonomy_from_header("hit key=100 other") == 100
    assert paired._taxonomy_from_header("hit key=-7") == -7
    assert paired._taxonomy_from_header("hit without a key") == -1
    assert paired._taxonomy_from_header("") == -1


def test_an_a3m_row_decodes_to_residue_codes_and_the_insertions_before_each_column():
    vocabulary = paired.protein_letter_to_res_type()

    residues, deletions = paired._decode_a3m_row("AcxC-Zq", 4, vocabulary)

    assert paired._emitted_length("AcxC-Zq") == 4
    assert residues.tolist() == [ALANINE, CYSTEINE, MSA_GAP_TOKEN_ID, PROTEIN_UNK_RES_TYPE]
    assert deletions.tolist() == [0.0, 2.0, 0.0, 0.0]
    assert paired._decode_a3m_row("ACDEACDE", 2, vocabulary)[0].tolist() == [ALANINE, CYSTEINE]


def test_an_alignment_decodes_to_residue_and_deletion_matrices():
    msa = MSA([FastaEntry("q", "ACDE"), FastaEntry("h", "AcCDE")])

    residues, deletions = paired.msa_to_res_type_and_deletions(msa, paired.protein_letter_to_res_type())  # (m, l) each

    assert residues.shape == (2, 4) and deletions.tolist() == [[0, 0, 0, 0], [0, 1, 0, 0]]
    assert residues[1].tolist() == [ALANINE, CYSTEINE, ASPARTATE, GLUTAMATE]


def construct(chain_msas, max_pairs=8192, max_seqs=16384):
    queries = {0: np.array([ALANINE, CYSTEINE]), 1: np.array([ASPARTATE, GLUTAMATE])}  # (l,) per chain
    return paired.construct_paired_msa(
        chain_msas,
        queries,
        token_asym_ids=np.array([0, 0, 1, 1]),
        token_res_ids=np.array([0, 1, 0, 1]),
        max_pairs=max_pairs,
        max_seqs=max_seqs,
    )  # (m, t) each


def test_rows_that_share_a_taxonomy_across_chains_are_paired_and_the_rest_follow_unpaired():
    first = MSA([FastaEntry("q0", "AC"), FastaEntry("a key=100", "AD"), FastaEntry("b key=200", "EC")])
    second = MSA([FastaEntry("q1", "DE"), FastaEntry("c key=100", "DD"), FastaEntry("d key=300", "EE")])
    vocabulary = paired.protein_letter_to_res_type()

    residues, deletions, paired_mask = construct({0: first, 1: second})  # (m, t) each

    assert residues.shape == (3, 4) and deletions.shape == (3, 4)
    assert residues[0].tolist() == [ALANINE, CYSTEINE, ASPARTATE, GLUTAMATE]  # the queries
    assert residues[1].tolist() == [vocabulary[letter] for letter in "ADDD"]  # the key=100 rows of both chains
    assert residues[2].tolist() == [vocabulary[letter] for letter in "ECEE"]  # the leftover rows, unpaired
    assert paired_mask.tolist() == [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]


def test_a_chain_without_an_alignment_contributes_only_its_query_and_gaps_elsewhere():
    first = MSA([FastaEntry("q0", "AC"), FastaEntry("a key=100", "AD")])

    residues, _, paired_mask = construct({0: first, 1: None})  # (m, t) each

    assert residues.shape == (2, 4)
    assert residues[0].tolist() == [ALANINE, CYSTEINE, ASPARTATE, GLUTAMATE]
    assert residues[1].tolist() == [ALANINE, ASPARTATE, MSA_GAP_TOKEN_ID, MSA_GAP_TOKEN_ID]
    assert paired_mask[1].tolist() == [0, 0, 0, 0]


def test_the_pairing_limits_cap_the_rows_and_a_chain_with_no_tokens_is_skipped():
    first = MSA([FastaEntry("q0", "AC"), FastaEntry("a key=100", "AD"), FastaEntry("b key=100", "EC")])
    second = MSA([FastaEntry("q1", "DE"), FastaEntry("c key=100", "DD"), FastaEntry("d key=100", "EE")])

    capped, _, _ = construct({0: first, 1: second}, max_pairs=2)
    truncated, _, _ = construct({0: first, 1: second}, max_seqs=1)
    unused_chain, _, _ = paired.construct_paired_msa(
        {0: first, 1: second},
        {0: np.array([ALANINE, CYSTEINE]), 1: np.array([ASPARTATE, GLUTAMATE])},
        token_asym_ids=np.array([0, 0]),
        token_res_ids=np.array([0, 1]),
    )

    assert capped.shape[0] == 2 and truncated.shape[0] == 1
    assert unused_chain.shape == (3, 2)
