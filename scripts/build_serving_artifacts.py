"""Create deterministic serving manifest without training or test labels."""

import argparse
from pathlib import Path

from recsys.serving.artifacts import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="artifacts/video_games_2018/serving_manifest.json")
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    write_manifest(args.output, Path(args.root), args.version)
    print(args.output)


if __name__ == "__main__":
    main()
