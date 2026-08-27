"""Amazon Reviews 2023 input schemas."""

REVIEW_REQUIRED_FIELDS = {"user_id", "parent_asin", "timestamp", "rating"}
METADATA_REQUIRED_FIELDS = {"parent_asin", "title", "main_category"}


def missing_fields(record: dict[str, object], required: set[str]) -> list[str]:
    """Return required field names absent from a JSON record."""
    return sorted(required.difference(record))

