"""Simple SQLite storage for baselines and recent time-series.

Stores baseline serialized dicts in a table; stores recent deviation time-series optionally.
"""
import sqlite3
import json
from typing import Optional


class Storage:
    def __init__(self, path: str = "gaitguard.db"):
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self._ensure()

    def _ensure(self):
        cur = self.conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS baselines (worker_id INTEGER PRIMARY KEY, json TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS series (worker_id INTEGER, timestamp REAL, deviation REAL)""")
        self.conn.commit()

    def save_baseline(self, worker_id: int, data: dict):
        j = json.dumps(data)
        cur = self.conn.cursor()
        cur.execute("REPLACE INTO baselines (worker_id, json) VALUES (?,?)", (worker_id, j))
        self.conn.commit()

    def load_baseline(self, worker_id: int) -> Optional[dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT json FROM baselines WHERE worker_id = ?", (worker_id,))
        row = cur.fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def append_series(self, worker_id: int, timestamp: float, deviation: float):
        cur = self.conn.cursor()
        cur.execute("INSERT INTO series (worker_id, timestamp, deviation) VALUES (?,?,?)", (worker_id, float(timestamp), float(deviation)))
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
