"""Public archive confirmation parsing does not need a network connection."""

import pytest

from tools.confidence.workflow import ARCHIVE_COPIES, ARCHIVES, DriveConfirmation, prepare_data


def test_drive_confirmation_extracts_public_download_fields():
    parser = DriveConfirmation()
    parser.feed(
        '<form id="download-form" action="https://drive.usercontent.google.com/download">'
        '<input type="hidden" name="id" value="archive-id">'
        '<input type="hidden" name="confirm" value="t">'
        '<input type="hidden" name="uuid" value="confirmation-id">'
        "</form>"
    )
    assert parser.action == "https://drive.usercontent.google.com/download"
    assert parser.fields == {"id": "archive-id", "confirm": "t", "uuid": "confirmation-id"}


def test_drive_error_page_does_not_supply_a_download_action():
    parser = DriveConfirmation()
    parser.feed("<html><title>Download quota exceeded</title></html>")
    assert parser.action == ""


def test_user_copy_keeps_original_partial_separate(monkeypatch, tmp_path):
    from tools.confidence import data, workflow

    archives = tmp_path / "archives"
    archives.mkdir()
    partial = archives / "rcsb_multimer.tar.zst.partial"
    partial.write_bytes(b"original partial")
    downloaded = []

    def download(identifier, destination):
        downloaded.append((identifier, destination.name))
        destination.write_bytes(b"copied archive")

    monkeypatch.setattr(workflow, "download_archive", download)
    monkeypatch.setattr(data, "safe_extract_archive", lambda *args, **kwargs: None)
    report = prepare_data(
        tmp_path, source="rcsb_multimer", drive_id=ARCHIVE_COPIES["rcsb_multimer"]
    )
    assert downloaded == [(ARCHIVE_COPIES["rcsb_multimer"], "rcsb_multimer-user-copy.tar.zst")]
    assert partial.read_bytes() == b"original partial"
    assert report["drive_id"] == ARCHIVE_COPIES["rcsb_multimer"]
    assert report["official_drive_id"] == ARCHIVES["rcsb_multimer"]
    assert len(report["sha256"]) == 64


def test_unapproved_archive_copy_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="user-supplied copy"):
        prepare_data(tmp_path, source="rcsb_multimer", drive_id="unapproved")
