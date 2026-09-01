"""Package ignored runtime artifacts for a GitHub Release attachment."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERSION = "recsys_phase6_content_v1"
DATA_FILES = [
    "data/processed/video_games_2018/interactions.parquet",
    "data/processed/video_games_2018/items.parquet",
    "data/processed/video_games_2018/items_content.parquet",
]


def package(version: str) -> tuple[Path, Path]:
    version_dir = ROOT / "artifacts" / "versions" / version
    manifest_path = version_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_files = [entry["path"] for entry in manifest["artifacts"].values()]
    files = [*artifact_files, *DATA_FILES, manifest_path.relative_to(ROOT).as_posix()]
    missing = [path for path in files if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing release files: {missing}")

    output_dir = ROOT / "dist"
    output_dir.mkdir(exist_ok=True)
    archive = output_dir / f"{version}_runtime.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for relative_path in sorted(set(files)):
            bundle.write(ROOT / relative_path, arcname=relative_path)

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = archive.with_suffix(".zip.sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    return archive, checksum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    args = parser.parse_args()
    archive, checksum = package(args.version)
    print(json.dumps({"archive": str(archive), "checksum": str(checksum)}, indent=2))


if __name__ == "__main__":
    main()
