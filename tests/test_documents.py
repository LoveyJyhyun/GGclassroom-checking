"""Reading the formats students actually hand in."""
import io
import tarfile
import zipfile

from checker import documents

DOC_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    "<w:body>"
    "<w:p><w:r><w:t>Câu 1: 42</w:t></w:r></w:p>"
    "<w:p></w:p><w:p></w:p><w:p></w:p>"
    "<w:p><w:r><w:t>Câu 2: 3,14</w:t></w:r><w:r><w:tab/><w:t>(sai số 0,01)</w:t></w:r></w:p>"
    "<w:p><w:r><w:t>a &amp; b</w:t><w:br/><w:t>second line</w:t></w:r></w:p>"
    "</w:body></w:document>"
)


def make_docx(document_xml: str = DOC_XML) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document_xml)
    return buffer.getvalue()


def make_xlsx() -> bytes:
    shared = ('<sst><si><t>Câu</t></si><si><t>Đáp án</t></si>'
              "<si><t>mức thu nhập</t></si></sst>")
    sheet = ("<worksheet><sheetData>"
             '<row><c t="s"><v>0</v></c><c t="s"><v>1</v></c></row>'
             '<row><c><v>1</v></c><c><v>42</v></c></row>'
             '<row><c><v>2</v></c><c t="inlineStr"><is><t>3,14</t></is></c></row>'
             "<row></row>"
             "</sheetData></worksheet>")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("xl/workbook.xml", "<workbook/>")
        archive.writestr("xl/sharedStrings.xml", shared)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)
    return buffer.getvalue()


def make_pptx() -> bytes:
    def slide(text):
        return f"<p:sld><p:cSld><a:t>{text}</a:t></p:cSld></p:sld>"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("ppt/presentation.xml", "<presentation/>")
        archive.writestr("ppt/slides/slide1.xml", slide("Bài 1"))
        archive.writestr("ppt/slides/slide10.xml", slide("Bài 10"))
        archive.writestr("ppt/slides/slide2.xml", slide("Bài 2"))
    return buffer.getvalue()


def make_zip(files: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, body in files.items():
            archive.writestr(name, body)
    return buffer.getvalue()


def make_tar(files: dict) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, body in files.items():
            raw = body if isinstance(body, bytes) else body.encode()
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    return buffer.getvalue()


# --- Office formats ---------------------------------------------------------

def test_docx_text():
    text = documents.docx_text(make_docx())
    assert "Câu 1: 42" in text
    assert "Câu 2: 3,14\t(sai số 0,01)" in text
    assert "a & b" in text and "second line" in text
    assert "<w:" not in text
    assert "\n\n\n" not in text                  # Word's empty paragraphs collapsed


def test_xlsx_text_resolves_shared_and_inline_strings():
    text = documents.xlsx_text(make_xlsx())
    rows = text.splitlines()
    assert rows[0] == "Câu\tĐáp án"               # shared strings looked up
    assert rows[1] == "1\t42"
    assert rows[2] == "2\t3,14"                   # inline string
    assert len(rows) == 3                         # the empty row is dropped


def test_pptx_text_is_in_slide_order():
    assert documents.pptx_text(make_pptx()).split("\n\n") == ["Bài 1", "Bài 2", "Bài 10"]


# --- Identifying files ------------------------------------------------------

def test_sniff_uses_the_bytes_not_the_name():
    assert documents.sniff("mystery", make_docx()) == "docx"
    assert documents.sniff("mystery", make_xlsx()) == "xlsx"
    assert documents.sniff("mystery", make_pptx()) == "pptx"
    assert documents.sniff("mystery", make_zip({"a.txt": "hi"})) == "zip"
    assert documents.sniff("x", b"%PDF-1.7") == "pdf"
    assert documents.sniff("x", b"Rar!\x1a\x07\x00") == "rar"
    assert documents.sniff("x", b"\x89PNG\r\n") == "image"
    assert documents.sniff("x", b"\xd0\xcf\x11\xe0") == "legacy-office"
    assert documents.sniff("x", "1: 42".encode()) == "text"
    assert documents.sniff("x", b"\x00\x01\x02\xff") == "unknown"


def test_a_file_with_no_extension_is_still_read():
    """Five students submitted files with no extension at all."""
    out = documents.extract("Nguyen Van A - bai 1", make_docx())
    assert "Câu 1: 42" in out["parts"][0][1]


# --- Archives ---------------------------------------------------------------

def test_zip_yields_one_part_per_readable_file():
    raw = make_zip({
        "bai/solution.py": "print(42)\n",
        "bai/traloi.docx": make_docx(),
        "bai/so_lieu.xlsx": make_xlsx(),
        "bai/plot.png": b"\x89PNG rubbish",
        "__MACOSX/._junk": b"junk",
        "bai/": b"",
    })
    out = documents.extract("BaiTap.zip", raw)
    names = [name for name, _ in out["parts"]]
    assert names == ["bai/solution.py", "bai/traloi.docx", "bai/so_lieu.xlsx"]
    assert dict(out["parts"])["bai/solution.py"] == "print(42)\n"
    assert "Câu 1: 42" in dict(out["parts"])["bai/traloi.docx"]
    assert "skipped" in out["note"]                 # the .png and the macOS junk


def test_tar_is_read_too():
    out = documents.extract("bai.tar", make_tar({"answer.txt": "1: 42\n"}))
    assert out["parts"] == [("answer.txt", "1: 42\n")]


def test_zip_with_nothing_readable_says_so():
    out = documents.extract("anh.zip", make_zip({"a.png": b"\x89PNG", "b.jpg": b"\xff\xd8\xff"}))
    assert out["parts"] == []
    assert "Nothing readable" in out["note"] and "2 file(s) skipped" in out["note"]


def test_member_count_is_capped(monkeypatch):
    monkeypatch.setattr(documents, "MAX_MEMBERS", 3)
    out = documents.extract("many.zip", make_zip({f"f{i}.txt": f"line {i}" for i in range(10)}))
    assert len(out["parts"]) == 3
    assert "skipped" in out["note"]


def test_a_huge_member_is_skipped_not_unpacked(monkeypatch):
    monkeypatch.setattr(documents, "MAX_UNPACKED_BYTES", 100)
    out = documents.extract("bomb.zip", make_zip({"big.txt": "x" * 5000, "small.txt": "1: 42"}))
    assert [n for n, _ in out["parts"]] == ["small.txt"]


def test_a_broken_member_does_not_lose_the_rest():
    out = documents.extract("mixed.zip", make_zip({
        "broken.docx": b"PK\x03\x04 corrupt", "good.txt": "1: 42"}))
    assert [n for n, _ in out["parts"]] == ["good.txt"]


def test_corrupt_archive_reports_instead_of_raising():
    out = documents.extract("bad.zip", b"PK\x03\x04 then nonsense")
    assert out["parts"] == [] and "could not be" in out["note"].lower()


# --- Formats that genuinely cannot be read ----------------------------------

def test_unsupported_formats_explain_what_to_ask_for():
    assert "zip instead" in documents.extract("x.rar", b"Rar!\x1a\x07\x00")["note"]
    assert "Google Doc" in documents.extract("x.pdf", b"%PDF-1.7")["note"]
    assert "text recognition" in documents.extract("x.png", b"\x89PNG\r\n")["note"]
    assert ".docx" in documents.extract("x.doc", b"\xd0\xcf\x11\xe0")["note"]
    for name, raw in [("x.rar", b"Rar!"), ("x.pdf", b"%PDF"), ("x.png", b"\x89PNG")]:
        assert documents.extract(name, raw)["parts"] == []


def test_an_unreadable_submission_scores_nothing_not_zero():
    """We could not open the file, so nothing is known - 0% would be a false mark."""
    from checker import results
    report = {
        "students": {
            "ok": {"name": "An", "notes": [], "files": ["bai.docx"]},
            "rar": {"name": "Chi", "notes": ["bai.rar: ..."], "files": []},
        },
        "numbers": {"key": {"1": {"value": 42}}, "results": {
            "ok": {"results": [{"question": "1", "ok": True, "got": 42}],
                   "correct": 1, "total": 1},
            "rar": {"results": [{"question": "1", "ok": False, "got": None}],
                    "correct": 0, "total": 1},
        }},
    }
    assert results.score(report, "ok")["percent"] == 100.0
    assert results.score(report, "rar") is None
    row = [r for r in results.to_csv(report).splitlines() if r.startswith("Chi")][0]
    assert "NOTHING READABLE" in row
    assert ",0.0," not in row and ",0," not in row
