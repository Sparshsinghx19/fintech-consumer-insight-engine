"""Tests for upload filename sanitization and safe saving."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion import sanitize_upload_filename, save_uploaded_pdf


@pytest.mark.parametrize(
    "name, expected",
    [
        ("report.pdf", "report.pdf"),
        ("Consumer Survey (2024).PDF", "Consumer Survey (2024).pdf"),
        ("../../etc/evil.pdf", "evil.pdf"),
        ("..\\..\\Windows\\evil.pdf", "evil.pdf"),
        ("C:\\Users\\me\\report.pdf", "report.pdf"),
        ("/abs/path/report.pdf", "report.pdf"),
        ("rep<o>rt|?.pdf", "rep_o_rt_.pdf"),
        ("x" * 300 + ".pdf", "x" * 146 + ".pdf"),
    ],
)
def test_sanitize_upload_filename(name, expected):
    assert sanitize_upload_filename(name) == expected


@pytest.mark.parametrize("name", ["", "notes.txt", "report", "..pdf", "../.pdf", "???.pdf", "report.pdf.exe"])
def test_sanitize_upload_filename_rejects_unusable_names(name):
    assert sanitize_upload_filename(name) is None


def test_save_uploaded_pdf_stays_inside_data_dir(tmp_path: Path):
    data_dir = tmp_path / "raw"

    dest = save_uploaded_pdf(data_dir, "../../escape.pdf", b"%PDF-1.4 test")

    assert dest == (data_dir / "escape.pdf").resolve()
    assert dest.read_bytes() == b"%PDF-1.4 test"
    assert not (tmp_path / "escape.pdf").exists()
    assert sorted(p.name for p in data_dir.iterdir()) == ["escape.pdf"]  # no temp file left behind


def test_save_uploaded_pdf_rejects_invalid_name(tmp_path: Path):
    with pytest.raises(ValueError):
        save_uploaded_pdf(tmp_path, "payload.exe", b"data")
    assert not any(tmp_path.iterdir())
