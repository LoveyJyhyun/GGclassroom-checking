"""Turning a Drive attachment into text, and uploading an answer sheet."""
import pytest

from checker import answerkey, classroom, documents
from test_documents import make_docx, make_xlsx, make_zip


class _Exec:
    def __init__(self, value):
        self._value = value

    def execute(self):
        return self._value


class FakeDrive:
    """Minimal stand-in for the Drive API client."""

    def __init__(self, meta, body):
        self._meta, self._body = meta, body

    def files(self):
        return self

    def get(self, fileId=None, fields=None):
        return _Exec(self._meta)

    def get_media(self, fileId=None):
        return _Exec(self._body)

    def export(self, fileId=None, mimeType=None):
        return _Exec(self._body)


def drive_with(meta, body):
    cr = classroom.Classroom.__new__(classroom.Classroom)   # skip the API build()
    cr.drive = FakeDrive(meta, body)
    return cr


# --- reading attachments ----------------------------------------------------

def test_reads_a_docx_attachment():
    meta = {"mimeType": classroom.DOCX_MIME, "name": "bai lam.docx", "size": "5000"}
    info = drive_with(meta, make_docx()).file_content("f1")
    assert info["note"] == ""
    assert "Câu 1: 42" in info["text"]
    assert info["parts"] == [("bai lam.docx", info["text"])]


def test_reads_a_zip_as_separate_parts():
    """15 students submitted .zip; each file inside must come back on its own."""
    raw = make_zip({"bai/main.py": "print(1)\n", "bai/traloi.docx": make_docx()})
    meta = {"mimeType": "application/x-zip-compressed", "name": "bai.zip", "size": "900"}
    info = drive_with(meta, raw).file_content("f1")
    assert [n for n, _ in info["parts"]] == ["bai/main.py", "bai/traloi.docx"]
    assert "print(1)" in info["text"] and "Câu 1: 42" in info["text"]


def test_reads_a_file_with_no_extension_by_sniffing_it():
    meta = {"mimeType": "application/octet-stream", "name": "Nguyen Van A - bai 1",
            "size": "900"}
    info = drive_with(meta, make_docx()).file_content("f1")
    assert "Câu 1: 42" in info["text"]


def test_exported_google_doc_still_works():
    meta = {"mimeType": "application/vnd.google-apps.document", "name": "Essay", "size": "0"}
    info = drive_with(meta, "1: 42\n").file_content("f1")
    assert info["text"] == "1: 42\n" and info["parts"] == [("Essay", "1: 42\n")]


def test_refuses_a_rar_with_advice():
    meta = {"mimeType": "application/x-rar-compressed", "name": "bai.rar", "size": "100"}
    info = drive_with(meta, b"Rar!\x1a\x07\x00").file_content("f1")
    assert info["text"] is None and "zip instead" in info["note"]


def test_oversized_document_is_not_downloaded():
    meta = {"mimeType": classroom.DOCX_MIME, "name": "huge.docx",
            "size": str(documents.MAX_DOCUMENT_BYTES + 1)}
    info = drive_with(meta, b"").file_content("f1")
    assert info["text"] is None and "too large" in info["note"]


def test_an_archive_may_be_bigger_than_a_document():
    """Only the text members are unpacked, so the archive limit is higher."""
    meta = {"mimeType": "application/zip", "name": "bai.zip",
            "size": str(documents.MAX_DOCUMENT_BYTES + 1)}
    assert drive_with(meta, make_zip({"a.txt": "1: 42"})).file_content("f1")["text"] == "1: 42"


def test_an_archive_over_its_own_limit_is_refused():
    meta = {"mimeType": "application/zip", "name": "bai.zip",
            "size": str(documents.MAX_ARCHIVE_BYTES + 1)}
    info = drive_with(meta, make_zip({"a.txt": "x"})).file_content("f1")
    assert info["text"] is None and "too large" in info["note"]


# --- uploading an answer sheet ----------------------------------------------

def test_docx_answer_sheet_round_trips_into_a_key():
    from checker import numbers
    key = numbers.parse_key(answerkey.read_upload("dapan.docx", make_docx()))
    assert key["1"]["value"] == 42
    assert key["2"]["value"] == 3.14 and key["2"]["tol"] == "0,01"


def test_an_answer_sheet_may_be_an_xlsx_or_a_zip():
    assert "Đáp án" in answerkey.read_upload("dapan.xlsx", make_xlsx())
    bundle = make_zip({"dapan/cau.txt": "1: 42\n", "dapan/anh.png": b"\x89PNG"})
    assert "1: 42" in answerkey.read_upload("dapan.zip", bundle)


def test_damaged_upload_is_rejected_with_a_message():
    with pytest.raises(ValueError, match="damaged"):
        answerkey.read_upload("dapan.docx", b"PK\x03\x04 corrupt")
    with pytest.raises(ValueError, match="could not be read"):
        answerkey.read_upload("dapan.bin", b"\x00\x01\x02\xff")


def test_pdf_doc_and_rar_uploads_are_refused_with_advice():
    with pytest.raises(ValueError, match="Google Doc"):
        answerkey.read_upload("dapan.pdf", b"%PDF-1.4")
    with pytest.raises(ValueError, match=r"\.docx"):
        answerkey.read_upload("dapan.doc", b"\xd0\xcf\x11\xe0")
    with pytest.raises(ValueError, match="zip instead"):
        answerkey.read_upload("dapan.rar", b"Rar!\x1a\x07\x00")
