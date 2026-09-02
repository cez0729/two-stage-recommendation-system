from recsys.ranking.pipeline import (
    BASE_FEATURE_COLUMNS,
    CONTENT_FEATURE_COLUMNS,
    feature_columns_for_config,
    fuse_candidates,
)


def test_ranker_feature_schema_is_config_aware() -> None:
    baseline = feature_columns_for_config({"retrieval": {}})
    content = feature_columns_for_config({"retrieval": {"content": {"enabled": True}}})

    assert baseline == BASE_FEATURE_COLUMNS
    assert content == CONTENT_FEATURE_COLUMNS
    assert len(baseline) == 16
    assert len(content) == 18
    assert content[:2] == baseline[:2]
    assert content[4:] == baseline[2:]


def test_fuse_candidates_deduplicates_and_rewards_multiple_sources() -> None:
    fused = fuse_candidates(
        [(1, 0.9), (2, 0.8)],
        [(2, 0.7), (3, 0.6)],
        [(2, 3.0), (4, 2.0)],
        pool_size=4,
        rrf_constant=60,
    )

    assert [row["item_idx"] for row in fused] == [2, 1, 3, 4]
    assert fused[0]["source_count"] == 3
    assert len({row["item_idx"] for row in fused}) == 4
