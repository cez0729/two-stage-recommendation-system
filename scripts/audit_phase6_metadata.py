"""Run the Phase 6 metadata gate and persist approved static content fields."""

from __future__ import annotations

import argparse

import pandas as pd

from recsys.config import load_yaml
from recsys.data.metadata import audit_metadata, extract_catalog_metadata
from recsys.utils.io import write_json


def run(config_path: str) -> dict[str, object]:
    config = load_yaml(config_path)
    data = config["data"]
    items = pd.read_parquet(data["items_path"])
    interactions = pd.read_parquet(data["interactions_path"])
    content_items, duplicate_records = extract_catalog_metadata(data["raw_metadata_path"], items)
    audit = audit_metadata(
        content_items,
        interactions,
        duplicate_records=duplicate_records,
    )
    content_items.to_parquet(data["content_items_path"], index=False)
    write_json(config["outputs"]["metadata_audit"], audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase6_content.yaml")
    args = parser.parse_args()
    audit = run(args.config)
    print(f"Metadata decision: {audit['decision']}")
    print(f"Catalog match rate: {audit['raw_match_rate']:.2%}")
    print(f"Strict-cold targets: {audit['strict_cold']['test_targets']}")


if __name__ == "__main__":
    main()
