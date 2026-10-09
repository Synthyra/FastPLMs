"""Reading, counting, appending, and writing FASTA records for ESMFold2, by path, gzip file, or open stream."""

import gzip
import io
import pytest

from fastplms.models.esmfold2 import esmfold2_parsing as parsing
from fastplms.models.esmfold2.esmfold2_parsing import FastaEntry


RECORDS = [("first record", "ACDE"), ("second record", "FGHI")]
TEXT = ">first record\nACDE\n>second record\nFGHI"


def test_records_are_written_without_a_trailing_newline_to_a_path_or_a_stream(tmp_path):
    destination = tmp_path / "nested" / "records.fasta"
    stream = io.StringIO()

    parsing.write_sequences(RECORDS, destination)
    parsing.write_sequences(iter(RECORDS), stream)
    parsing.write_sequences([], tmp_path / "empty.fasta")

    assert destination.read_text(encoding="utf-8") == TEXT
    assert stream.getvalue() == TEXT and not stream.closed
    assert (tmp_path / "empty.fasta").read_text(encoding="utf-8") == ""


def test_the_first_record_comes_from_a_path_a_compressed_file_or_a_stream_that_stays_open(tmp_path):
    plain = tmp_path / "records.fasta"
    plain.write_text(TEXT, encoding="utf-8")
    compressed = tmp_path / "records.fasta.gz"
    with gzip.open(compressed, mode="wt", encoding="utf-8") as handle:
        handle.write(TEXT)
    stream = io.StringIO(TEXT)

    assert parsing.read_first_sequence(plain) == FastaEntry("first record", "ACDE")
    assert parsing.read_first_sequence(compressed) == FastaEntry("first record", "ACDE")
    assert parsing.read_first_sequence(stream) == FastaEntry("first record", "ACDE")
    assert not stream.closed
    assert list(parsing.read_sequences(plain)) == [FastaEntry(*record) for record in RECORDS]


def test_a_file_without_records_is_refused(tmp_path):
    empty = tmp_path / "empty.fasta"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Found no sequences in input"):
        parsing.read_first_sequence(empty)


def test_headers_are_counted_without_parsing_and_a_missing_file_has_none(tmp_path):
    path = tmp_path / "records.fasta"
    path.write_text(TEXT + "\n>third\nKLMN\n", encoding="utf-8")

    assert parsing.count_fasta_sequences(path) == 3
    assert parsing.count_fasta_sequences(str(path)) == 3
    assert parsing.count_fasta_sequences(tmp_path / "missing.fasta") == 0


def test_a_record_is_appended_after_a_newline_even_when_the_file_lacks_one(tmp_path):
    path = tmp_path / "deeper" / "records.fasta"

    parsing.append_fasta_sequence("first", "ACDE", path)
    parsing.append_fasta_sequence("second", "FGHI", str(path))
    path.write_text(path.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")
    parsing.append_fasta_sequence("third", "KLMN", path)

    assert path.read_text(encoding="utf-8") == ">first\nACDE\n>second\nFGHI\n>third\nKLMN\n"
    assert parsing.count_fasta_sequences(path) == 3
