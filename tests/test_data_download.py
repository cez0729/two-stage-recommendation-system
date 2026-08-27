import gzip
import json

from recsys.data.download import audit_jsonl_gz
from recsys.data.schemas import REVIEW_REQUIRED_FIELDS


def test_audit_jsonl_gz_reports_schema(tmp_path) -> None:
    path = tmp_path / "reviews.jsonl.gz"
    records = [
        {"user_id": "u1", "parent_asin": "i1", "timestamp": 1, "rating": 5.0},
        {"user_id": "u2", "parent_asin": "i2", "timestamp": 2, "rating": 4.0},
    ]
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")

    audit = audit_jsonl_gz(path, REVIEW_REQUIRED_FIELDS)

    assert audit["rows_scanned"] == 2
    assert audit["missing_required_rows"] == 0

