"""Hard rules a notification must pass before it can be sent. Any failure blocks the send."""
from __future__ import annotations

import re

_QUOTED = re.compile(r'"([^"]{3,120})"')
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
_MONEY = re.compile(r"\$\s?\d")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def check(msg: dict, recs: list[dict]) -> list[str]:
    errors = []
    if not isinstance(msg, dict) or not msg.get("body") or not msg.get("subject"):
        return ["malformed_message"]
    allowed_ids = {r["job_id"] for r in recs}
    allowed_titles = {_norm(r["title"]) for r in recs}
    ids = msg.get("job_ids") or []
    if not ids:
        errors.append("no_job_ids")
    if any(i not in allowed_ids for i in ids):
        errors.append("mentions_unrecommended_job_id")
    for q in _QUOTED.findall(msg["subject"] + " " + msg["body"]):
        if _norm(q) not in allowed_titles:
            errors.append(f"mentions_unrecommended_title:{q[:40]}")
    if len(msg["subject"]) > 90:
        errors.append("subject_too_long")
    if len(msg["body"]) > 600:
        errors.append("body_too_long")
    text = msg["subject"] + " " + msg["body"]
    for r in recs:                     # recommended titles/ids may contain digit runs (XING/OLX "job <id>")
        text = text.replace(f'"{r["title"]}"', " ")
        text = re.sub(rf"\b{int(r['job_id'])}\b", " ", text) if len(str(r["job_id"])) >= 7 else text
    if _EMAIL.search(text) or _PHONE.search(text):
        errors.append("contains_contact_info")
    if _MONEY.search(text):
        errors.append("states_pay_not_in_data")
    return errors
