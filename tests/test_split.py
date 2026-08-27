import pandas as pd

from recsys.data.split import chronological_split, validate_split


def test_four_way_chronological_split() -> None:
    frame = pd.DataFrame(
        {
            "user_idx": [0] * 5 + [1] * 6,
            "item_idx": list(range(5)) + list(range(10, 16)),
            "timestamp": pd.to_datetime(
                list(range(1, 6)) + list(range(1, 7)), unit="D", utc=True
            ),
        }
    )

    split = chronological_split(frame)
    stats = validate_split(split)

    assert split[split["user_idx"] == 0]["split"].tolist() == [
        "retrieval_train",
        "retrieval_train",
        "rank_train",
        "validation",
        "test",
    ]
    assert stats["strict_time_order"] is True
    assert stats["target_items_absent_from_history"] is True


def test_day_level_ties_use_distinct_target_timestamps() -> None:
    frame = pd.DataFrame(
        {
            "user_idx": [0] * 7,
            "item_idx": [1, 2, 3, 4, 5, 6, 7],
            "timestamp": pd.to_datetime([1, 2, 3, 3, 4, 4, 5], unit="D", utc=True),
        }
    )

    split = chronological_split(
        frame, same_timestamp_policy="select_one_per_target_timestamp"
    )
    stats = validate_split(split)

    assert split["split"].tolist() == [
        "retrieval_train",
        "retrieval_train",
        "rank_train",
        "validation",
        "test",
    ]
    assert split.attrs["split_diagnostics"]["ambiguous_target_rows_dropped"] == 2
    assert stats["strict_time_order"] is True
