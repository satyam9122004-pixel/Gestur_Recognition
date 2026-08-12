"""SQLite-backed storage for registered people and their feature vectors."""
import sqlite3
import json
import os


SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    vector TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES people (id) ON DELETE CASCADE
);
"""


class Database:
    def __init__(self, db_path="database/people.db"):
        self.db_path = db_path
        dirname = os.path.dirname(db_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def add_person(self, name):
        cur = self.conn.execute("INSERT INTO people (name) VALUES (?)", (name,))
        self.conn.commit()
        return cur.lastrowid

    def add_feature_vector(self, person_id, vector):
        vector_json = json.dumps(list(map(float, vector)))
        self.conn.execute(
            "INSERT INTO features (person_id, vector) VALUES (?, ?)",
            (person_id, vector_json),
        )
        self.conn.commit()

    def get_person_by_name(self, name):
        cur = self.conn.execute("SELECT id, name FROM people WHERE name = ?", (name,))
        row = cur.fetchone()
        return {"id": row[0], "name": row[1]} if row else None

    def get_all_people(self):
        cur = self.conn.execute("SELECT id, name FROM people")
        return [{"id": r[0], "name": r[1]} for r in cur.fetchall()]

    def get_feature_vectors(self, person_id=None):
        """Return list of (person_id, name, vector) for all stored features,
        optionally filtered to a single person."""
        query = (
            "SELECT f.person_id, p.name, f.vector FROM features f "
            "JOIN people p ON p.id = f.person_id"
        )
        params = ()
        if person_id is not None:
            query += " WHERE f.person_id = ?"
            params = (person_id,)

        cur = self.conn.execute(query, params)
        results = []
        for pid, name, vector_json in cur.fetchall():
            results.append((pid, name, json.loads(vector_json)))
        return results

    def delete_person(self, person_id):
        self.conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
        self.conn.commit()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
