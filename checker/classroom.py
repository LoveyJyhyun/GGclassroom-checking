"""Thin wrapper around the Google Classroom and Drive APIs."""
from __future__ import annotations

import os

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from . import documents

COURSEWORK_READ = "https://www.googleapis.com/auth/classroom.coursework.students.readonly"
# The only write permission the app can ask for: draft grades on student work.
# Off unless GRADE_WRITE=1, and even then grades are saved only when the teacher
# approves them on the report page.
COURSEWORK_WRITE = "https://www.googleapis.com/auth/classroom.coursework.students"
GRADE_WRITE = os.environ.get("GRADE_WRITE") == "1"

SCOPES = [
    "https://www.googleapis.com/auth/classroom.courses.readonly",
    "https://www.googleapis.com/auth/classroom.rosters.readonly",
    "https://www.googleapis.com/auth/classroom.profile.emails",
    COURSEWORK_WRITE if GRADE_WRITE else COURSEWORK_READ,
    # Needed to read the contents of files students attach.
    "https://www.googleapis.com/auth/drive.readonly",
]


def can_write_grades(scopes) -> bool:
    """Whether a signed-in session was granted the draft-grade permission."""
    return COURSEWORK_WRITE in (scopes or [])

TEXT_EXPORTS = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.presentation": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
}
RAW_TEXT_MIMES = {
    "application/json", "application/javascript", "application/x-python",
    "application/x-ipynb+json", "application/vnd.google.colaboratory",
}
CODE_EXTENSIONS = (".py", ".ipynb", ".js", ".txt", ".csv", ".md", ".json")
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# Anything not exported by Google and not plain text is handed to documents.extract,
# which decides from the bytes what it is. Archives may be bigger than documents
# because only their text members are read.
ARCHIVE_HINTS = (".zip", ".rar", ".tar", ".tar.gz", ".tgz", ".7z")

# Kept for callers that only want Word text.
docx_text = documents.docx_text


def _paginate(request_fn, key: str, **kwargs) -> list[dict]:
    items, token = [], None
    while True:
        resp = request_fn(pageToken=token, **kwargs).execute()
        items.extend(resp.get(key, []))
        token = resp.get("nextPageToken")
        if not token:
            return items


class Classroom:
    def __init__(self, credentials):
        self.api = build("classroom", "v1", credentials=credentials, cache_discovery=False)
        self.drive = build("drive", "v3", credentials=credentials, cache_discovery=False)

    def courses(self) -> list[dict]:
        return _paginate(self.api.courses().list, "courses",
                         teacherId="me", courseStates=["ACTIVE"])

    def course(self, course_id: str) -> dict:
        return self.api.courses().get(id=course_id).execute()

    def students(self, course_id: str) -> list[dict]:
        students = _paginate(self.api.courses().students().list, "students", courseId=course_id)
        students.sort(key=lambda s: s["profile"]["name"].get("fullName", "").lower())
        return students

    def student(self, course_id: str, user_id: str) -> dict:
        return self.api.courses().students().get(courseId=course_id, userId=user_id).execute()

    def course_work(self, course_id: str) -> list[dict]:
        return _paginate(self.api.courses().courseWork().list, "courseWork",
                         courseId=course_id, courseWorkStates=["PUBLISHED"])

    def one_course_work(self, course_id: str, work_id: str) -> dict:
        return self.api.courses().courseWork().get(courseId=course_id, id=work_id).execute()

    def submissions(self, course_id: str, work_id: str = "-") -> list[dict]:
        """All submissions for one coursework item, or every item when work_id is '-'."""
        return _paginate(self.api.courses().courseWork().studentSubmissions().list,
                         "studentSubmissions", courseId=course_id, courseWorkId=work_id)

    def submission(self, course_id: str, work_id: str, sub_id: str) -> dict:
        return (self.api.courses().courseWork().studentSubmissions()
                .get(courseId=course_id, courseWorkId=work_id, id=sub_id).execute())

    def set_draft_grade(self, course_id: str, work_id: str, sub_id: str, grade: float) -> dict:
        """Save a draft grade (visible to the teacher only until they return the work).

        Google allows this only for coursework created by this app's Cloud project;
        for anything else it answers 403, which the caller reports per student.
        """
        return (self.api.courses().courseWork().studentSubmissions()
                .patch(courseId=course_id, courseWorkId=work_id, id=sub_id,
                       updateMask="draftGrade", body={"draftGrade": grade}).execute())

    def file_text(self, file_id: str) -> tuple[str | None, str]:
        """Return (text, note). text is None when the file can't be read as text."""
        info = self.file_content(file_id)
        return info["text"], info["note"]

    def file_content(self, file_id: str) -> dict:
        """Return {name, mime, text, parts, note}; text is None when unreadable.

        `parts` is [(member name, text)] - one entry for a document, one per readable
        file for an archive, so the caller can route each member by its own name.
        """
        result = {"name": "", "mime": "", "text": None, "parts": [], "note": ""}
        try:
            meta = self.drive.files().get(fileId=file_id, fields="mimeType,size,name").execute()
        except HttpError as e:
            result["note"] = f"Cannot open file ({e.status_code})."
            return result
        mime, name = meta.get("mimeType", ""), meta.get("name", "")
        result.update(name=name, mime=mime)
        size = int(meta.get("size", 0))
        looks_archived = name.lower().endswith(ARCHIVE_HINTS)
        limit = documents.MAX_ARCHIVE_BYTES if looks_archived else documents.MAX_DOCUMENT_BYTES
        try:
            if mime in TEXT_EXPORTS:
                data = self.drive.files().export(fileId=file_id,
                                                 mimeType=TEXT_EXPORTS[mime]).execute()
                if isinstance(data, bytes):
                    data = data.decode("utf-8", errors="replace")
                result["text"] = data.lstrip("\ufeff")
                result["parts"] = [(name, result["text"])]
                return result

            if size > limit:
                result["note"] = "File is too large to read."
                return result
            raw = self.drive.files().get_media(fileId=file_id).execute()
        except HttpError as e:
            result["note"] = f"Cannot read file ({e.status_code})."
            return result

        if not isinstance(raw, bytes):                 # a stub or an already-text body
            raw = str(raw).encode("utf-8", errors="replace")
        found = documents.extract(name, raw)
        result["parts"] = found["parts"]
        result["note"] = found["note"]
        if found["parts"]:
            result["text"] = "\n\n".join(text for _, text in found["parts"])
        return result
