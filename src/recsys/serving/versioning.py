"""Resolve an optional promoted serving version without changing model logic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_serving_config(config: dict[str, Any], root: str | Path = ".") -> dict[str, Any]:
    """Apply the promoted version pointer when it exists; otherwise keep config unchanged."""
    serving = config.get("serving", {})
    pointer_path = Path(root) / serving.get("current_pointer", "artifacts/versions/current.json")
    if not pointer_path.is_file():
        return config
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    manifest_path = Path(root) / pointer["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved = dict(config)
    resolved_serving = dict(resolved["serving"])
    resolved_serving.update(manifest.get("serving", {}))
    resolved_serving["version"] = str(pointer["version"])
    resolved["serving"] = resolved_serving
    return resolved
