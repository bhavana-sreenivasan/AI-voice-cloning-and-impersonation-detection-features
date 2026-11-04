"""SQLite record of every analysed call (number, time, risk, scores) + complaint status.

Also stores a speaker embedding per call so a new call can be matched against past calls
(e.g. the same scammer voice calling from different numbers).
"""
import json
import re
import sqlite3
import time

import numpy as np

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
    report_ref TEXT DEFAULT '',
    embedding TEXT,
    emb_model TEXT
)
"""


def _conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.execute(_SCHEMA)
    # Migration: add the embedding column to an older calls.db that doesn't have it yet.
    try:
        c.execute("ALTER TABLE calls ADD COLUMN embedding TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists
    # Migration: tag which speaker model made each embedding. Embeddings from different models
    # (e.g. ECAPA-TDNN vs ECAPA2) are both 192-d but NOT comparable, so rows without a tag
    # (made before this column existed) are treated as 'speechbrain' and never matched against ECAPA2.
    try:
        c.execute("ALTER TABLE calls ADD COLUMN emb_model TEXT")
        c.execute("UPDATE calls SET emb_model='speechbrain' WHERE embedding IS NOT NULL")
    except sqlite3.OperationalError:
        pass
    return c


def normalize_number(raw):
    """'+91 98765-43210' -> '+919876543210'. Blank -> 'unknown'. Invalid -> None."""
    raw = (raw or "").strip()
    if not raw:
        return "unknown"
    cleaned = re.sub(r"[\s\-().]", "", raw)
    return cleaned if re.fullmatch(r"\+?\d{7,15}", cleaned) else None


def log_call(phone, source, risk, label=None, p_spoof=None, identity_name=None, identity_score=None,
             identity_decision=None, reason="", audio_sha1="", verification=None, notes="",
             embedding=None, emb_model=None):
    """Insert one analysed call. `embedding` is an optional 1-D speaker embedding (numpy array/list);
    pass `emb_model=speaker.EMB_MODEL` with it so it is only ever compared with the same model's embeddings."""
    emb_json = None
    if embedding is not None:
        emb_json = json.dumps([float(x) for x in np.asarray(embedding, dtype=float).ravel()])
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO calls (ts, phone, source, risk, label, p_spoof, identity_name, identity_score,"
            " identity_decision, reason, audio_sha1, verification, notes, embedding, emb_model)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), phone, source, risk, label, p_spoof, identity_name, identity_score,
             identity_decision, reason, audio_sha1, verification, notes, emb_json,
             emb_model if embedding is not None else None))
        return cur.lastrowid


def similar_calls(emb, threshold=0.6, limit=5, exclude_id=None, emb_model=None):
    """Past calls whose speaker embedding is close to `emb` (cosine similarity >= threshold).

    Call this BEFORE log_call for the current clip (or pass exclude_id) so a call can't match itself.
    Returns a list of (similarity, call_id, phone, risk), best match first.
    If `emb_model` is given, only embeddings made by that model are considered (the 0.6 default is
    not calibrated for ECAPA2 either).
    """
    emb = np.asarray(emb, dtype=float).ravel()
    emb_norm = np.linalg.norm(emb)
    if emb_norm == 0:
        return []
    out = []
    with _conn() as c:
        for r in c.execute("SELECT id, phone, risk, embedding, emb_model FROM calls WHERE embedding IS NOT NULL"):
            if emb_model is not None and r["emb_model"] != emb_model:
                continue
            if exclude_id is not None and r["id"] == exclude_id:
                continue
            e = np.array(json.loads(r["embedding"]), dtype=float)
            if e.shape != emb.shape:
                continue
            denom = emb_norm * np.linalg.norm(e)
            if denom == 0:
                continue
            sim = float(np.dot(emb, e) / denom)
            if sim >= threshold:
                out.append((sim, r["id"], r["phone"], r["risk"]))
    return sorted(out, reverse=True)[:limit]


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
    """Most recent calls. The embedding column is left out to keep the table readable."""
    with _conn() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM calls ORDER BY ts DESC LIMIT ?", (limit,))]
    for r in rows:
        r.pop("embedding", None)
    return rows


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
