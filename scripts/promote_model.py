"""Promote a versioned serving snapshot only when regression gates pass."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from recsys.utils.io import write_json

ROOT = Path(__file__).resolve().parents[1]


def promote(version: str, pointer_path: str | Path = "artifacts/versions/current.json") -> dict:
    manifest_path = ROOT / "artifacts" / "versions" / version / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Version manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    acceptance = manifest.get("promotion", {}).get("acceptance", {})
    if not acceptance or not all(bool(value) for value in acceptance.values()):
        raise ValueError(f"Promotion gate failed for {version}: {acceptance}")
    pointer = ROOT / pointer_path
    pointer.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer.with_suffix(pointer.suffix + ".tmp")
    payload = {
        "version": version,
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
    }
    write_json(temporary, payload)
    os.replace(temporary, pointer)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--pointer", default="artifacts/versions/current.json")
    args = parser.parse_args()
    print(promote(args.version, args.pointer))


if __name__ == "__main__":
    main()
