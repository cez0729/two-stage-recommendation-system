from recsys.retrieval.candidate_filter import Candidate, CandidateFilter


def test_filter_removes_seen_invalid_duplicates_and_backfills() -> None:
    fallback = [Candidate(3, 1.0, "popularity"), Candidate(4, 0.5, "popularity")]
    candidate_filter = CandidateFilter({1, 2, 3, 4}, fallback)
    candidates = [
        Candidate(1, 5.0, "model"),
        Candidate(2, 4.0, "model"),
        Candidate(2, 3.0, "model"),
        Candidate(99, 2.0, "model"),
    ]

    result = candidate_filter.apply(candidates, seen_items={1}, k=3)

    assert [candidate.item_idx for candidate in result] == [2, 3, 4]
    assert len({candidate.item_idx for candidate in result}) == len(result)

