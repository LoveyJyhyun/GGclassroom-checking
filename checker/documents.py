"""Pull text out of the files students actually hand in, using only the stdlib.

Word, Excel and PowerPoint files are zips of XML, and `.zip`/`.tar` submissions are
just more files inside. All of that is reachable with `zipfile`/`tarfile` + regex,
so it is read rather than reported as "not supported".

`.rar` needs a proprietary unpacker and `.pdf`/images need a PDF parser or OCR;
those get a note telling the teacher what to ask for instead.

Every reader returns *parts*: a list of (member name, text). A plain document has
one part; an archive has one per readable file inside, so the caller can route each
member on its own name (a `.py` inside a zip still reaches the code check).
"""
from __future__ import annotations

import html
import io
import re
import tarfile
import zipfile

# An archive is only read for its text members, so it may be larger than a document.
MAX_DOCUMENT_BYTES = 2_000_000
MAX_ARCHIVE_BYTES = 20_000_000
MAX_MEMBERS = 40
MAX_UNPACKED_BYTES = 8_000_000          # guards against a zip bomb

TEXT_MEMBER_EXTENSIONS = (".txt", ".md", ".csv", ".py", ".js", ".ipynb", ".json",
                          ".html", ".rst", ".r", ".sql", ".java", ".c", ".cpp")
SKIP_MEMBER = re.compile(r"(^|/)(__MACOSX|\.|~\$)|/$")

UNSUPPORTED = {
    "rar": "RAR archives need WinRAR to unpack; ask for a .zip instead.",
    "pdf": "PDF text is not read; ask for a Google Doc or a .docx.",
    "image": "Images are not read (no text recognition); ask for a Google Doc or a .docx.",
    "legacy-office": "This is an old .doc/.xls; ask for .docx/.xlsx or a Google Doc.",
    "broken-zip": "This file starts like a zip/Office file but could not be opened; "
                  "it may have been damaged on upload. Ask for it again.",
}


def _strip_tags(xml: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", xml))


# --- Office formats ---------------------------------------------------------

def docx_text(raw: bytes) -> str:
    """The visible text of a Word document."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", errors="replace")
    # Paragraph and row ends become newlines, tabs stay tabs, then tags go away.
    xml = re.sub(r"<w:(?:br|cr)\b[^>]*/?>", "\n", xml)
    xml = re.sub(r"<w:tab\b[^>]*/?>", "\t", xml)
    xml = re.sub(r"</w:(?:p|tr)>", "\n", xml)
    return re.sub(r"\n{3,}", "\n\n", _strip_tags(xml)).strip()


def xlsx_text(raw: bytes) -> str:
    """Cell values of every sheet, one row per line, tab separated."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            xml = archive.read("xl/sharedStrings.xml").decode("utf-8", errors="replace")
            shared = [_strip_tags(si) for si in re.findall(r"<si>(.*?)</si>", xml, re.DOTALL)]

        lines = []
        sheets = sorted(n for n in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
        for sheet in sheets:
            xml = archive.read(sheet).decode("utf-8", errors="replace")
            for row in re.findall(r"<row\b[^>]*>(.*?)</row>", xml, re.DOTALL):
                # ponytail: cell references are ignored, so empty columns do not keep
                # their position. Fine for reading answers, wrong for rebuilding a grid.
                cells = []
                for cell in re.findall(r"<c\b[^>]*/>|<c\b[^>]*>.*?</c>", row, re.DOTALL):
                    inline = re.search(r"<is>(.*?)</is>", cell, re.DOTALL)
                    value = re.search(r"<v>(.*?)</v>", cell, re.DOTALL)
                    if inline:
                        cells.append(_strip_tags(inline.group(1)))
                    elif value and 't="s"' in cell:
                        index = int(value.group(1)) if value.group(1).isdigit() else -1
                        cells.append(shared[index] if 0 <= index < len(shared) else "")
                    elif value:
                        cells.append(html.unescape(value.group(1)))
                if any(c.strip() for c in cells):
                    lines.append("\t".join(cells))
    return "\n".join(lines)


def pptx_text(raw: bytes) -> str:
    """Slide text, in slide order, one blank line between slides."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        slides = sorted(
            (n for n in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
            key=lambda n: int(re.search(r"(\d+)", n).group(1)))
        out = []
        for slide in slides:
            xml = archive.read(slide).decode("utf-8", errors="replace")
            runs = [_strip_tags(t) for t in re.findall(r"<a:t>(.*?)</a:t>", xml, re.DOTALL)]
            if any(r.strip() for r in runs):
                out.append("\n".join(runs))
    return "\n\n".join(out)


# --- What is this file? -----------------------------------------------------

def _zip_kind(raw: bytes) -> str:
    """Tell docx / xlsx / pptx / plain zip apart by what is inside."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = set(archive.namelist())
    except (zipfile.BadZipFile, OSError):
        return "broken-zip"
    if "word/document.xml" in names:
        return "docx"
    if any(n.startswith("xl/workbook") for n in names):
        return "xlsx"
    if any(n.startswith("ppt/presentation") for n in names):
        return "pptx"
    return "zip"


def sniff(name: str, raw: bytes) -> str:
    """Identify a file from its bytes, falling back to its extension."""
    if raw.startswith(b"PK\x03\x04"):
        return _zip_kind(raw)
    if raw.startswith(b"%PDF"):
        return "pdf"
    if raw.startswith(b"Rar!"):
        return "rar"
    if raw.startswith((b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"BM")) or raw[:4] == b"RIFF":
        return "image"
    if raw.startswith(b"\xd0\xcf\x11\xe0"):
        return "legacy-office"
    if raw.startswith((b"\x1f\x8b", b"BZh")) or name.lower().endswith((".tar", ".tar.gz", ".tgz")):
        return "tar"
    try:
        raw.decode("utf-8")
        return "text"
    except UnicodeDecodeError:
        return "unknown"


# --- Archives ---------------------------------------------------------------

def _readable_member(name: str) -> str | None:
    """Which reader a file inside an archive needs, or None to skip it."""
    if SKIP_MEMBER.search(name):
        return None
    lower = name.lower()
    if lower.endswith(".docx"):
        return "docx"
    if lower.endswith(".xlsx"):
        return "xlsx"
    if lower.endswith(".pptx"):
        return "pptx"
    if lower.endswith(TEXT_MEMBER_EXTENSIONS):
        return "text"
    return None


def _member_text(kind: str, data: bytes) -> str:
    if kind == "docx":
        return docx_text(data)
    if kind == "xlsx":
        return xlsx_text(data)
    if kind == "pptx":
        return pptx_text(data)
    return data.decode("utf-8", errors="replace").lstrip("﻿")


def archive_parts(raw: bytes, kind: str) -> tuple[list[tuple[str, str]], str]:
    """Read the text files inside a .zip or .tar. Nested archives are not opened.

    Members are listed first and only the readable ones are decompressed, so a zip
    full of screenshots costs nothing to look through.
    """
    parts, skipped, budget = [], 0, MAX_UNPACKED_BYTES
    try:
        with (tarfile.open(fileobj=io.BytesIO(raw)) if kind == "tar"
              else zipfile.ZipFile(io.BytesIO(raw))) as archive:
            if kind == "tar":
                listing = [(m.name, m.size, m) for m in archive.getmembers() if m.isfile()]
                read = lambda m: (archive.extractfile(m) or io.BytesIO(b"")).read()
            else:
                listing = [(i.filename, i.file_size, i) for i in archive.infolist()
                           if not i.is_dir()]
                read = archive.read

            for member_name, size, handle in listing:
                member_kind = _readable_member(member_name)
                if member_kind is None or len(parts) >= MAX_MEMBERS or size > budget:
                    skipped += 1
                    continue
                budget -= size
                try:
                    text = _member_text(member_kind, read(handle))
                except Exception:           # noqa: BLE001 - one bad member, keep the rest
                    skipped += 1
                    continue
                if text.strip():
                    parts.append((member_name, text))
                else:
                    skipped += 1
    except (zipfile.BadZipFile, tarfile.TarError, OSError, EOFError) as e:
        return [], f"This archive could not be opened ({type(e).__name__})."

    if not parts:
        return [], ("Nothing readable inside this archive"
                    + (f" ({skipped} file(s) skipped)." if skipped else "."))
    note = f"{skipped} file(s) inside were skipped." if skipped else ""
    return parts, note


# --- One entry point --------------------------------------------------------

def extract(name: str, raw: bytes) -> dict:
    """Return {"parts": [(member name, text)], "note": str} for any attachment."""
    kind = sniff(name, raw)

    if kind in UNSUPPORTED:
        return {"parts": [], "note": UNSUPPORTED[kind]}
    if kind in ("zip", "tar"):
        parts, note = archive_parts(raw, kind)
        return {"parts": parts, "note": note}
    if kind in ("docx", "xlsx", "pptx"):
        try:
            text = _member_text(kind, raw)
        except Exception as e:              # noqa: BLE001 - malformed Office file
            return {"parts": [], "note": f"This {kind} could not be read ({type(e).__name__})."}
        return ({"parts": [(name, text)], "note": ""} if text.strip()
                else {"parts": [], "note": f"This {kind} has no text in it."})
    if kind == "text":
        text = raw.decode("utf-8", errors="replace").lstrip("﻿")
        return {"parts": [(name, text)], "note": ""}
    return {"parts": [], "note": "This file is not text and its format is not recognised."}


def demo() -> None:
    """Self-check: a zip of mixed files comes back as separate routable parts."""
    doc = io.BytesIO()
    with zipfile.ZipFile(doc, "w") as z:
        z.writestr("word/document.xml",
                   '<w:document xmlns:w="x"><w:p><w:r><w:t>Câu 1: 42</w:t></w:r></w:p>'
                   "</w:document>")
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w") as z:
        z.writestr("bai/solution.py", "print(42)\n")
        z.writestr("bai/traloi.docx", doc.getvalue())
        z.writestr("bai/plot.png", b"\x89PNG not read")
        z.writestr("__MACOSX/._junk", b"junk")

    out = extract("BaiTap.zip", bundle.getvalue())
    names = [n for n, _ in out["parts"]]
    assert names == ["bai/solution.py", "bai/traloi.docx"], names
    assert dict(out["parts"])["bai/solution.py"] == "print(42)\n"
    assert "Câu 1: 42" in dict(out["parts"])["bai/traloi.docx"]
    assert "skipped" in out["note"]                       # the .png and the junk

    assert extract("x.rar", b"Rar!\x1a\x07\x00")["note"].startswith("RAR")
    assert extract("x.pdf", b"%PDF-1.7")["note"].startswith("PDF")
    assert extract("no-extension", b"1: 42\n")["parts"] == [("no-extension", "1: 42\n")]
    print("OK: archives split into routable parts, unsupported formats explain themselves")


if __name__ == "__main__":
    demo()
