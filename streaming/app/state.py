from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


class StateStore:
    """One short transaction per write; usable by a future API while training runs."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL,
                    actor TEXT NOT NULL, status TEXT NOT NULL,
                    config TEXT NOT NULL, started TEXT NOT NULL, ended TEXT,
                    UNIQUE(run_id, actor));
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, run INTEGER NOT NULL REFERENCES runs(id),
                    time TEXT NOT NULL, kind TEXT NOT NULL, round_id INTEGER,
                    details TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def begin(self, run_id, actor, config):
        with self.connect() as db:
            cursor = db.execute(
                'INSERT INTO runs(run_id,actor,status,config,started) VALUES(?,?,?,?,?)',
                (run_id, actor, 'Running', json.dumps(config, sort_keys=True), now()))
            return cursor.lastrowid

    def interrupt_previous(self):
        # Call only after obtaining the exclusive runtime lock.
        with self.connect() as db:
            db.execute("UPDATE runs SET status='Interrupted', ended=? WHERE status='Running'", (now(),))

    def emit(self, run, kind, round_id=None, **details):
        encoded = json.dumps(details, sort_keys=True, allow_nan=False)
        with self.connect() as db:
            db.execute('INSERT INTO events(run,time,kind,round_id,details) VALUES(?,?,?,?,?)',
                       (run, now(), kind, round_id, encoded))

    def finish(self, run, status):
        if status not in {'Completed', 'Failed', 'Interrupted'}:
            raise ValueError('Invalid terminal status')
        with self.connect() as db:
            db.execute('UPDATE runs SET status=?,ended=? WHERE id=?', (status, now(), run))

    def runs(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute('SELECT * FROM runs ORDER BY id DESC')]

    def events(self, run, after=0, limit=500):
        if not 1 <= limit <= 1000:
            raise ValueError('limit must be between 1 and 1000')
        with self.connect() as db:
            rows = db.execute('SELECT * FROM events WHERE run=? AND id>? ORDER BY id LIMIT ?',
                              (run, after, limit))
            return [{**dict(row), 'details': json.loads(row['details'])} for row in rows]
