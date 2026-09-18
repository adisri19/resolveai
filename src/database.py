"""SQLite access. stdlib sqlite3 only -- the schema is three tables, an ORM earns nothing here."""
import os
import sqlite3
from contextlib import contextmanager

import shutil
from pathlib import Path

def get_db_path() -> str:
    if os.getenv("VERCEL"):
        tmp_db = Path("/tmp/app.db")
        if not tmp_db.exists():
            seed_db = Path(__file__).resolve().parent.parent / "data" / "seed.db"
            if seed_db.exists():
                try:
                    shutil.copy2(seed_db, tmp_db)
                except Exception as exc:
                    print(f"[warn] failed to copy seed.db: {exc}")
        return str(tmp_db)
    return os.getenv("DB_PATH", "app.db")

DB_PATH = get_db_path()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS tickets (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER NOT NULL REFERENCES users(id),
    message             TEXT    NOT NULL,
    order_value_inr     REAL,
    days_since_delivery INTEGER,
    days_since_dispatch INTEGER,
    product_type        TEXT,
    opened_status       TEXT,
    order_status        TEXT,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS decisions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id  INTEGER NOT NULL REFERENCES tickets(id),
    action     TEXT    NOT NULL,
    reason     TEXT    NOT NULL,
    confidence REAL    NOT NULL,
    sources    TEXT    NOT NULL,          -- JSON array of knowledge-base filenames
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Cached policy-chunk embeddings. Keyed by content hash so re-ingesting is free.
CREATE TABLE IF NOT EXISTS kb_chunks (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    source    TEXT NOT NULL,
    text      TEXT NOT NULL,
    hash      TEXT NOT NULL UNIQUE,
    embedding BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tickets_user ON tickets(user_id);
CREATE INDEX IF NOT EXISTS idx_decisions_ticket ON decisions(ticket_id);
"""


@contextmanager
def connect():
    """One connection per operation. SQLite is fine with this and it sidesteps
    the thread-affinity problem you get sharing a connection across FastAPI workers."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with connect() as conn:
        conn.executescript(SCHEMA)
