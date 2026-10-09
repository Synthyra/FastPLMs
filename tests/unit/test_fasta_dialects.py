"""The three FASTA readers share one scanner and keep their own rules.

The ``reference_*`` functions are the three readers as they were written before they shared the
scanner: ``iter_fasta`` in embeddings/inputs.py, ``read_fasta_sequences`` in models/e1/retrieval.py and
``parse_fasta`` in models/esmfold2/esmfold2_parsing.py. Every input must give each new reader the
same records, or the same exception type and message, as its reference.
"""

from __future__ import annotations

import random
import pytest

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from fastplms.embeddings.inputs import (
    EMBEDDING_FASTA,
    FastaDialect,
    FastaRecord,
    iter_fasta,
    parse_fasta,
    scan_fasta_lines,
)
from fastplms.models.e1.retrieval import E1_FASTA, read_fasta_sequences
from fastplms.models.esmfold2.esmfold2_parsing import parse_fasta as parse_fasta_text


def reference_embedding_fasta(lines: Iterable[str], path: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    identifier: str | None = None
    sequence_parts: list[str] = []
    found_record = False
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                found_record = True
                records.append((identifier, "".join(sequence_parts)))
            identifier = line[1:].strip().split(maxsplit=1)[0]
            if not identifier:
                raise ValueError(f"Missing FASTA identifier on line {line_number}.")
            sequence_parts = []
        else:
            if identifier is None:
                raise ValueError(
                    f"Sequence data precedes the first FASTA header on line {line_number}."
                )
            sequence_parts.append("".join(line.split()))
    if identifier is not None:
        found_record = True
        records.append((identifier, "".join(sequence_parts)))
    if not found_record:
        raise ValueError(f"No FASTA records found in {path}.")
    return records


def reference_e1_fasta(lines: Iterable[str], path: str) -> dict[str, str]:
    sequences: dict[str, str] = {}
    header: str | None = None
    parts: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                sequences[header] = "".join(parts)
            header = line[1:].strip()
            parts = []
        else:
            if header is None:
                raise ValueError(f"FASTA sequence found before header in {path}")
            parts.append(line)
    if header is not None:
        sequences[header] = "".join(parts)
    return sequences


def reference_esmfold2_fasta(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence_lines: list[str] = []
    found_record = False
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            if header is not None:
                found_record = True
                records.append((header, "".join(sequence_lines)))
            header = line[1:].strip()
            sequence_lines.clear()
        elif header is not None:
            sequence_lines.append(line)
    if header is not None:
        found_record = True
        records.append((header, "".join(sequence_lines)))
    if not found_record:
        raise ValueError("Found no sequences in input")
    return records


Outcome = tuple[str, Any]


def outcome(read: Callable[[], Any]) -> Outcome:
    try:
        return ("records", read())
    except Exception as error:  # noqa: broad-except  the exception type and message are the compared outcome
        return ("error", (type(error), str(error)))


FRAGMENTS = (
    ">a",
    ">a description words",
    ">b\tTabbed",
    ">",
    ">   ",
    "  >indented header",
    "# comment",
    "#>not a header",
    "",
    "   ",
    "\t",
    "ACDEFG",
    "AC DE\tFG",
    "  ACD  ",
    "ACD-.*",
    "ac\x00gt",
    ">a",
    "\x0b",
    "\N{LINE SEPARATOR}",
)
LINE_ENDINGS = ("\n", "\r\n", "\r")


def corpus() -> list[str]:
    texts = [
        "",
        "\n",
        ">only\n",
        ">x\nAA\n",
        "AA\n>x\nCC\n",
        "# note\n>x\nAA\n",
        ">x\nAA\n>x\nCC\n",
        ">x desc\nAC\nGT\n\n>y\n\nTT",
        ">\nAA\n",
        "  \n>x\n  AC  \n",
    ]
    generator = random.Random(20261005)
    for _ in range(400):
        count = generator.randint(0, 9)
        pieces = [generator.choice(FRAGMENTS) + generator.choice(LINE_ENDINGS) for _ in range(count)]
        texts.append("".join(pieces))
    return texts


CORPUS = corpus()
CORPUS_IDS = [f"case{index}" for index in range(len(CORPUS))]


@pytest.mark.parametrize("text", CORPUS, ids=CORPUS_IDS)
def test_embedding_dialect_matches_the_original_iter_fasta(text: str) -> None:
    lines = text.splitlines(keepends=True)
    expected = outcome(lambda: reference_embedding_fasta(lines, "input.fa"))
    observed = outcome(lambda: list(scan_fasta_lines(lines, EMBEDDING_FASTA, source="input.fa")))
    assert observed == (
        (expected[0], [FastaRecord(*record) for record in expected[1]])
        if expected[0] == "records"
        else expected
    )


@pytest.mark.parametrize("text", CORPUS, ids=CORPUS_IDS)
def test_e1_dialect_matches_the_original_read_fasta_sequences(text: str) -> None:
    lines = text.splitlines(keepends=True)
    expected = outcome(lambda: reference_e1_fasta(lines, "msa.a3m"))
    observed = outcome(
        lambda: {
            record.header: record.sequence
            for record in scan_fasta_lines(lines, E1_FASTA, source="msa.a3m")
        }
    )
    assert observed == expected
    if expected[0] == "records":
        assert list(observed[1]) == list(expected[1])


@pytest.mark.parametrize("text", CORPUS, ids=CORPUS_IDS)
def test_esmfold2_dialect_matches_the_original_parse_fasta(text: str) -> None:
    expected = outcome(lambda: reference_esmfold2_fasta(text))
    observed = outcome(lambda: [tuple(entry) for entry in parse_fasta_text(text)])
    assert observed == expected


def test_each_public_reader_reads_a_file_as_before(tmp_path: Path) -> None:
    text = ">p1 first protein\nAC D\n>p1\nGG\n\n>p2\nTT\n"
    path = tmp_path / "proteins.fasta"
    path.write_bytes(text.encode("utf-8"))
    assert [(r.id, r.sequence) for r in iter_fasta(path)] == [("p1", "ACD"), ("p1", "GG"), ("p2", "TT")]
    assert [(r.id, r.sequence) for r in parse_fasta(str(path))] == [("p1", "ACD"), ("p1", "GG"), ("p2", "TT")]
    assert read_fasta_sequences(str(path)) == {"p1 first protein": "AC D", "p1": "GG", "p2": "TT"}
    assert [tuple(entry) for entry in parse_fasta_text(text)] == [
        ("p1 first protein", "AC D"),
        ("p1", "GG"),
        ("p2", "TT"),
    ]


def test_each_reader_keeps_its_own_error_for_a_missing_header(tmp_path: Path) -> None:
    path = tmp_path / "orphan.fasta"
    path.write_text("ACDE\n>x\nGG\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"^Sequence data precedes the first FASTA header on line 1\.$"):
        list(iter_fasta(path))
    with pytest.raises(ValueError, match=r"^FASTA sequence found before header in .*orphan\.fasta$"):
        read_fasta_sequences(str(path))
    assert [tuple(entry) for entry in parse_fasta_text("ACDE\n>x\nGG\n")] == [("x", "GG")]


def test_empty_input_is_an_error_only_for_two_readers(tmp_path: Path) -> None:
    path = tmp_path / "empty.fasta"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"^No FASTA records found in .*empty\.fasta\.$"):
        list(iter_fasta(path))
    assert read_fasta_sequences(str(path)) == {}
    with pytest.raises(ValueError, match=r"^Found no sequences in input$"):
        list(parse_fasta_text(""))


def test_an_empty_header_has_no_identifier() -> None:
    # The embedding reader takes the first word of the header; a bare ">" has none.
    with pytest.raises(IndexError):
        list(scan_fasta_lines([">\n", "AA\n"], EMBEDDING_FASTA, source="input"))


def test_records_are_yielded_before_a_later_error_is_raised() -> None:
    dialect = FastaDialect(
        strip_lines=True,
        comment_prefix=None,
        first_word_header=False,
        squeeze_sequence_whitespace=False,
        orphan_message="orphan at {line_number} in {source}",
        empty_message=None,
    )
    scanner = scan_fasta_lines([">a\n", "AC\n"], dialect, source="stream")
    assert list(scanner) == [FastaRecord("a", "AC")]
    with pytest.raises(ValueError, match="^orphan at 1 in stream$"):
        list(scan_fasta_lines(["AC\n"], dialect, source="stream"))


def test_the_scanner_streams_one_line_at_a_time() -> None:
    consumed: list[int] = []

    def lines() -> Iterable[str]:
        for index, line in enumerate((">a\n", "AC\n", ">b\n", "GT\n", ">c\n", "TT\n")):
            consumed.append(index)
            yield line

    scanner = scan_fasta_lines(lines(), EMBEDDING_FASTA, source="stream")
    assert next(scanner) == FastaRecord("a", "AC")
    assert consumed == [0, 1, 2]
