"""Append-only local storage; every provider response is retained verbatim."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

from audit_llm_raters.schema import SCORE_COLUMNS, ScoreRecord


class ScoreStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scores (
                request_id TEXT PRIMARY KEY,
                essay_id TEXT NOT NULL,
                dataset TEXT NOT NULL,
                prompt_id TEXT NOT NULL,
                human_r1 REAL,
                human_r2 REAL,
                human_rater_ids TEXT NOT NULL,
                model TEXT NOT NULL,
                model_version TEXT NOT NULL,
                prompt_variant TEXT NOT NULL,
                rubric_variant TEXT NOT NULL,
                run_id INTEGER NOT NULL,
                raw_response TEXT NOT NULL,
                parsed_score INTEGER,
                parse_status TEXT NOT NULL,
                parse_detail TEXT NOT NULL,
                tokens_in INTEGER,
                tokens_out INTEGER,
                cost REAL,
                timestamp TEXT NOT NULL,
                scale_min INTEGER NOT NULL,
                scale_max INTEGER NOT NULL,
                ell INTEGER,
                error TEXT NOT NULL,
                latency_ms REAL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_scores_essay ON scores(dataset, essay_id)"
        )
        self.connection.commit()

    def completed_ids(self) -> set[str]:
        rows = self.connection.execute(
            "SELECT request_id FROM scores WHERE parse_status != 'api_error'"
        ).fetchall()
        return {row[0] for row in rows}

    def total_cost(self) -> float:
        value = self.connection.execute("SELECT COALESCE(SUM(cost), 0) FROM scores").fetchone()[0]
        return float(value)

    def add(self, record: ScoreRecord) -> None:
        values = record.as_dict()
        self.connection.execute(
            f"INSERT OR REPLACE INTO scores ({','.join(SCORE_COLUMNS)}) "
            f"VALUES ({','.join('?' for _ in SCORE_COLUMNS)})",
            [values[column] for column in SCORE_COLUMNS],
        )
        self.connection.commit()

    def add_many(self, records: Iterable[ScoreRecord]) -> None:
        rows = [record.as_dict() for record in records]
        if not rows:
            return
        self.connection.executemany(
            f"INSERT OR REPLACE INTO scores ({','.join(SCORE_COLUMNS)}) "
            f"VALUES ({','.join('?' for _ in SCORE_COLUMNS)})",
            [[row[column] for column in SCORE_COLUMNS] for row in rows],
        )
        self.connection.commit()

    def to_frame(self, include_api_errors: bool = True) -> pd.DataFrame:
        query = "SELECT * FROM scores"
        if not include_api_errors:
            query += " WHERE parse_status != 'api_error'"
        return pd.read_sql_query(query, self.connection)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ScoreStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
