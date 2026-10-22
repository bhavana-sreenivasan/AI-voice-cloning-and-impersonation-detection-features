"""SQLite record of every analysed call (number, time, risk, scores) + complaint status."""
import re
import sqlite3
import time

DB = "calls.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    phone TEXT NOT NULL,
    source TEXT NOT NULL,
    risk TEXT NOT NULL,
    label TEXT,
    p_spoof REAL,
    identity_name TEXT,
    identity_score REAL,
    identity_decision TEXT,
    reason TEXT,
    audio_sha1 TEXT,
    verification TEXT,
    notes TEXT DEFAULT '',
    reported INTEGER DEFAULT 0,
    report_ref TEXT DEFAULT ''
)
"""


def _conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.execute(_SCHEMA)
    return c


def normalize_number(raw):
    """'+91 98765-43210' -> '+919876543210'. Blank -> 'unknown'. Invalid -> None."""
    raw = (raw or "").strip()
    if not raw:
        return "unknown"
    cleaned = re.sub(r"[\s\-().]", "", raw)
    return cleaned if re.fullmatch(r"\+?\d{7,15}", cleaned) else None


def log_call(phone, source, risk, label=None, p_spoof=None, identity_name=None, identity_score=None,
             identity_decision=None, reason="", audio_sha1="", verification=None, notes=""):
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO calls (ts, phone, source, risk, label, p_spoof, identity_name, identity_score,"
            " identity_decision, reason, audio_sha1, verification, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), phone, source, risk, label, p_spoof, identity_name, identity_score,
             identity_decision, reason, audio_sha1, verification, notes))
        return cur.lastrowid


def update_verification(call_id, outcome, risk=None, reason=None):
    with _conn() as c:
        c.execute("UPDATE calls SET verification=? WHERE id=?", (outcome, call_id))
        if risk:
            c.execute("UPDATE calls SET risk=? WHERE id=?", (risk, call_id))
        if reason:
            c.execute("UPDATE calls SET reason=? WHERE id=?", (reason, call_id))


def mark_reported(call_id, ref=""):
    with _conn() as c:
        c.execute("UPDATE calls SET reported=1, report_ref=? WHERE id=?", (ref, call_id))


def get(call_id):
    with _conn() as c:
        row = c.execute("SELECT * FROM calls WHERE id=?", (call_id,)).fetchone()
        return dict(row) if row else None


def recent(limit=500):
    with _conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM calls ORDER BY ts DESC LIMIT ?", (limit,))]


def number_stats():
    """One row per number: total calls, high-risk calls, last seen, whether any was reported."""
    with _conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT phone, COUNT(*) AS calls, SUM(risk='High') AS high_risk, MAX(ts) AS last_seen,"
            " MAX(reported) AS reported FROM calls WHERE phone != 'unknown'"
            " GROUP BY phone ORDER BY high_risk DESC, calls DESC")]


def update_result(call_id, risk, reason, label=None, p_spoof=None, identity_name=None,
                  identity_score=None, identity_decision=None):
    """Refresh the analysis fields of an existing row (e.g. after the claimed speaker changed)."""
    with _conn() as c:
        c.execute("UPDATE calls SET risk=?, reason=?, label=?, p_spoof=?, identity_name=?,"
                  " identity_score=?, identity_decision=? WHERE id=?",
                  (risk, reason, label, p_spoof, identity_name, identity_score, identity_decision, call_id))
