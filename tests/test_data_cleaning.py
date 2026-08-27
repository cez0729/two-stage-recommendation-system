import pandas as pd

from recsys.data.clean import clean_interactions, iterative_k_core


def test_iterative_k_core_repeats_until_stable() -> None:
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3"],
            "item_id": ["i1", "i2", "i1", "i2", "i2"],
        }
    )

    filtered, iterations = iterative_k_core(frame, 2, 2)

    assert len(filtered) == 4
    assert set(filtered["user_id"]) == {"u1", "u2"}
    assert iterations[-1]["rows"] == 4


def test_clean_interactions_is_deterministic() -> None:
    raw = pd.DataFrame(
        {
            "user_id": ["u2"] * 5 + ["u1"] * 5,
            "parent_asin": [f"i{x}" for x in range(5)] * 2,
            "timestamp": list(range(1_000, 6_000, 1_000)) * 2,
            "rating": [5.0] * 10,
        }
    )
    config = {
        "project_seed": 42,
        "dataset": {
            "min_rating": 3.0,
            "deduplicate_user_item": True,
            "min_user_interactions": 5,
            "min_item_interactions": 2,
            "max_interactions": 100,
        },
    }

    cleaned, _ = clean_interactions(raw, config)

    assert cleaned.loc[cleaned["user_id"] == "u1", "user_idx"].unique().tolist() == [0]
    assert cleaned.loc[cleaned["item_id"] == "i0", "item_idx"].unique().tolist() == [0]

