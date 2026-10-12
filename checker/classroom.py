"""Thin wrapper around the Google Classroom and Drive APIs."""
from __future__ import annotations

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = [
    "https://www.googleapis.com/auth/classroom.courses.readonly",
    "https://www.googleapis.com/auth/classroom.rosters.readonly",
    "https://www.googleapis.com/auth/classroom.profile.emails",
    "https://www.googleapis.com/auth/classroom.coursework.students.readonly",
    # Needed to read the contents of files students attach.
    "https://www.googleapis.com/auth/drive.readonly",
]

MAX_TEXT_BYTES = 2_000_000
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

    def file_text(self, file_id: str) -> tuple[str | None, str]:
        """Return (text, note). text is None when the file can't be read as text."""
        info = self.file_content(file_id)
        return info["text"], info["note"]

    def file_content(self, file_id: str) -> dict:
        """Return {name, mime, text, note}; text is None when unreadable."""
        result = {"name": "", "mime": "", "text": None, "note": ""}
        try:
            meta = self.drive.files().get(fileId=file_id, fields="mimeType,size,name").execute()
        except HttpError as e:
            result["note"] = f"Cannot open file ({e.status_code})."
            return result
        mime, name = meta.get("mimeType", ""), meta.get("name", "")
        result.update(name=name, mime=mime)
        readable = (mime.startswith("text/") or mime in RAW_TEXT_MIMES
                    or name.lower().endswith(CODE_EXTENSIONS))
        try:
            if mime in TEXT_EXPORTS:
                data = self.drive.files().export(fileId=file_id, mimeType=TEXT_EXPORTS[mime]).execute()
            elif readable:
                if int(meta.get("size", 0)) > MAX_TEXT_BYTES:
                    result["note"] = "File is too large to read."
                    return result
                data = self.drive.files().get_media(fileId=file_id).execute()
            else:
                result["note"] = f"Reading {mime or 'this file type'} is not supported."
                return result
        except HttpError as e:
            result["note"] = f"Cannot read file ({e.status_code})."
            return result
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        result["text"] = data.lstrip("\ufeff")
        return result
