"""Persistent feedback event storage for the serving demo."""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

EVENT_TYPES = {"impression", "click", "purchase"}


class SQLiteFeedbackStore:
    """Small file-backed event store; each operation uses a short-lived connection."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=2.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_events (
                    event_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    event_type TEXT NOT NULL CHECK(
                        event_type IN ('impression', 'click', 'purchase')
                    ),
                    timestamp TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    rank_position INTEGER NOT NULL,
                    simulated INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def write(self, event: dict[str, Any]) -> bool:
        """Insert one event and return False for an existing event id."""
        event_type = str(event["event_type"])
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unsupported event_type: {event_type}")
        created_at = datetime.now().astimezone().isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO feedback_events
                (event_id, user_id, item_id, event_type, timestamp, request_id,
                 model_version, rank_position, simulated, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event["event_id"]),
                    str(event["user_id"]),
                    str(event["item_id"]),
                    event_type,
                    str(event["timestamp"]),
                    str(event["request_id"]),
                    str(event["model_version"]),
                    int(event["rank_position"]),
                    int(bool(event["simulated"])),
                    created_at,
                ),
            )
        return cursor.rowcount == 1

    def export(self, output_path: str | Path, output_format: str | None = None) -> int:
        """Export all events as JSONL or CSV and return the row count."""
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fmt = output_format or ("csv" if destination.suffix.lower() == ".csv" else "jsonl")
        with self._connect() as connection:
            rows = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM feedback_events ORDER BY created_at, event_id"
                )
            ]
        if fmt == "jsonl":
            with destination.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        elif fmt == "csv":
            fieldnames = [
                "event_id", "user_id", "item_id", "event_type", "timestamp",
                "request_id", "model_version", "rank_position", "simulated", "created_at",
            ]
            with destination.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        else:
            raise ValueError("output_format must be jsonl or csv")
        return len(rows)
