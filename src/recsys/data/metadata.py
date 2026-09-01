"""Deterministic Amazon metadata extraction and quality auditing."""

from __future__ import annotations

import gzip
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PLACEHOLDERS = {"", "unknown", "none", "n/a", "na", "null", "video games"}


def normalize_text(value: object) -> str:
    """Decode HTML entities and collapse whitespace in a metadata value."""
    return re.sub(r"\s+", " ", html.unescape(str(value or ""))).strip()


def normalize_brand(value: object) -> str:
    """Remove the common Amazon 'by' presentation prefix from brands."""
    return re.sub(r"^by\s+", "", normalize_text(value), flags=re.IGNORECASE).strip()


def normalize_category_path(value: object) -> str:
    """Convert a category list to a stable path while dropping HTML fragments."""
    if not isinstance(value, list):
        return ""
    parts = [normalize_text(part) for part in value]
    return " > ".join(part for part in parts if part and "<" not in part)


def is_usable(value: object) -> bool:
    """Return whether a cleaned field contains information beyond a placeholder."""
    return normalize_text(value).lower() not in PLACEHOLDERS


def extract_catalog_metadata(raw_path: str | Path, items: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Extract the first raw metadata record for every mapped item."""
    wanted = set(items["item_id"].astype(str))
    records: dict[str, dict[str, Any]] = {}
    duplicate_records = 0
    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            item_id = str(record.get("asin", ""))
            if item_id not in wanted:
                continue
            if item_id in records:
                duplicate_records += 1
                continue
            records[item_id] = record

    rows = []
    for item in items[["item_idx", "item_id"]].itertuples(index=False):
        record = records.get(str(item.item_id), {})
        rows.append(
            {
                "item_idx": int(item.item_idx),
                "item_id": str(item.item_id),
                "title": normalize_text(record.get("title")),
                "brand": normalize_brand(record.get("brand")),
                "main_category": normalize_text(record.get("main_cat")),
                "fine_category": normalize_category_path(record.get("category")),
                "metadata_matched": bool(record),
            }
        )
    return pd.DataFrame(rows), duplicate_records


def _field_summary(frame: pd.DataFrame, field: str) -> dict[str, Any]:
    values = frame[field].astype(str)
    usable = values[values.map(is_usable)]
    counts = Counter(usable)
    return {
        "usable_count": int(len(usable)),
        "coverage": float(len(usable) / len(frame)),
        "unique_values": int(usable.nunique()),
        "unique_ratio_among_usable": float(usable.nunique() / len(usable)) if len(usable) else 0.0,
        "median_characters": float(usable.str.len().median()) if len(usable) else 0.0,
        "top_values": [{"value": value, "count": count} for value, count in counts.most_common(10)],
    }


def audit_metadata(
    content_items: pd.DataFrame,
    interactions: pd.DataFrame,
    *,
    duplicate_records: int,
) -> dict[str, Any]:
    """Summarize catalog and strict-cold metadata coverage for the Phase 6 gate."""
    train_items = set(
        interactions.loc[interactions["split"] == "retrieval_train", "item_idx"].astype(int)
    )
    test = interactions[interactions["split"] == "test"]
    cold_targets = test[~test["item_idx"].astype(int).isin(train_items)]
    cold = cold_targets[["item_idx"]].merge(content_items, on="item_idx", how="left")
    fields = ["title", "brand", "main_category", "fine_category"]
    field_summaries = {field: _field_summary(content_items, field) for field in fields}
    cold_coverage = {
        field: float(cold[field].fillna("").map(is_usable).mean()) for field in fields
    }
    go = (
        field_summaries["title"]["coverage"] >= 0.95
        and field_summaries["fine_category"]["coverage"] >= 0.90
        and cold_coverage["title"] >= 0.95
    )
    return {
        "decision": "GO" if go else "NO_GO",
        "decision_rule": {
            "title_catalog_coverage_min": 0.95,
            "fine_category_catalog_coverage_min": 0.90,
            "strict_cold_title_coverage_min": 0.95,
        },
        "catalog_items": int(len(content_items)),
        "matched_raw_records": int(content_items["metadata_matched"].sum()),
        "raw_match_rate": float(content_items["metadata_matched"].mean()),
        "ignored_duplicate_raw_records": int(duplicate_records),
        "fields": field_summaries,
        "strict_cold": {
            "test_targets": int(len(cold)),
            "unique_items": int(cold["item_idx"].nunique()),
            "coverage": cold_coverage,
        },
        "excluded_fields": {
            "price": "unreliable parsing and possible temporal variation",
            "avg_rating": "absent from the formal source and temporally cumulative",
            "rating_number": "absent from the formal source and temporally cumulative",
        },
        "snapshot_limit": (
            "Metadata is a static public snapshot. Content fields do not use outcomes, but their "
            "historical availability time is not independently verified."
        ),
    }
