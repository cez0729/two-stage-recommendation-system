"""Shared candidate validation, seen-item filtering, and fallback logic."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """A scored item emitted by one retrieval source."""

    item_idx: int
    score: float
    source: str


class CandidateFilter:
    """Produce unique, valid, unseen candidates with deterministic backfill."""

    def __init__(self, valid_items: set[int], fallback: Sequence[Candidate]) -> None:
        self.valid_items = valid_items
        self.fallback = fallback

    def apply(
        self,
        candidates: Iterable[Candidate],
        *,
        seen_items: set[int],
        k: int,
    ) -> list[Candidate]:
        """Filter candidates, then append fallback items until k is reached."""
        output: list[Candidate] = []
        emitted: set[int] = set()
        for candidate in [*candidates, *self.fallback]:
            if candidate.item_idx not in self.valid_items:
                continue
            if candidate.item_idx in seen_items or candidate.item_idx in emitted:
                continue
            output.append(candidate)
            emitted.add(candidate.item_idx)
            if len(output) == k:
                break
        return output

