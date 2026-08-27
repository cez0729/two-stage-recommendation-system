"""Export persisted serving feedback for later offline training."""

import argparse

from recsys.serving.events import SQLiteFeedbackStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/feedback/events.db")
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=["jsonl", "csv"])
    args = parser.parse_args()
    count = SQLiteFeedbackStore(args.db).export(args.output, args.format)
    print(f"exported_events={count} output={args.output}")


if __name__ == "__main__":
    main()
