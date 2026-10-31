"""SQLite key-value cache for model calls and retrieval results."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path


def cache_key(**parts) -> str:
    """Hash of every component that determines an output (model, prompt, config, ...)."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Cache:
    """One connection shared by the runner's worker threads, serialised by a lock."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False, timeout=120)
        self.db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)")

    def get(self, key: str):
        with self.lock:
            row = self.db.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False)))
            self.db.commit()
