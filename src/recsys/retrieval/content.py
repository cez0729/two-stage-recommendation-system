"""Low-complexity content retrieval for cold and long-tail item analysis."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from recsys.retrieval.candidate_filter import Candidate


def content_document(row: pd.Series) -> str:
    """Build a field-aware document without outcome or post-interaction features."""
    title = str(row.get("title", ""))
    brand = str(row.get("brand", ""))
    category = str(row.get("fine_category", ""))
    return f"{title} brand_{brand.replace(' ', '_')} category_{category.replace(' ', '_')}"


class TfidfContentRecommender:
    """Recommend catalog items similar to the mean content vector of user history."""

    def __init__(
        self,
        *,
        max_features: int = 50_000,
        min_df: int = 2,
        ngram_max: int = 2,
        max_history: int = 50,
    ) -> None:
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_max = ngram_max
        self.max_history = max_history
        self.vectorizer: TfidfVectorizer | None = None
        self.item_matrix: sparse.csr_matrix | None = None
        self.item_ids = np.empty(0, dtype=np.int32)
        self.row_by_item: dict[int, int] = {}
        self.catalog: set[int] = set()

    def fit(self, items: pd.DataFrame, training_catalog: set[int]) -> TfidfContentRecommender:
        """Fit vocabulary on training items, then transform all metadata-visible items."""
        ordered = items.sort_values("item_idx", kind="stable").reset_index(drop=True)
        documents = ordered.apply(content_document, axis=1).tolist()
        train_mask = ordered["item_idx"].astype(int).isin(training_catalog).to_numpy()
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, self.ngram_max),
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=True,
            norm="l2",
        )
        training_documents = [
            document
            for document, keep in zip(documents, train_mask, strict=True)
            if keep
        ]
        vectorizer.fit(training_documents)
        matrix = vectorizer.transform(documents).tocsr()
        self.vectorizer = vectorizer
        self.item_matrix = matrix
        self.item_ids = ordered["item_idx"].to_numpy(dtype=np.int32)
        self.row_by_item = {int(item): row for row, item in enumerate(self.item_ids)}
        usable = ordered["title"].fillna("").astype(str).str.strip().ne("").to_numpy()
        self.catalog = set(map(int, self.item_ids[usable]))
        return self

    @classmethod
    def load_artifacts(
        cls,
        *,
        vectorizer_path: str | Path,
        item_matrix_path: str | Path,
        item_ids_path: str | Path,
        items: pd.DataFrame,
        max_history: int = 50,
    ) -> TfidfContentRecommender:
        """Load a frozen Phase 6 content index without refitting its vocabulary."""
        recommender = cls(max_history=max_history)
        recommender.vectorizer = joblib.load(vectorizer_path)
        recommender.item_matrix = sparse.load_npz(item_matrix_path).tocsr()
        recommender.item_ids = np.load(item_ids_path).astype(np.int32)
        if recommender.item_matrix.shape[0] != len(recommender.item_ids):
            raise ValueError("Content matrix and item ID artifact lengths differ")
        recommender.row_by_item = {
            int(item): row for row, item in enumerate(recommender.item_ids)
        }
        usable = items.set_index("item_idx").reindex(recommender.item_ids)["title"]
        recommender.catalog = set(
            map(int, recommender.item_ids[usable.fillna("").astype(str).str.strip().ne("")])
        )
        return recommender

    def _profile(self, history: Iterable[int]) -> sparse.csr_matrix | None:
        if self.item_matrix is None:
            raise RuntimeError("TfidfContentRecommender must be fitted before recommendation")
        rows = [
            self.row_by_item[item]
            for item in list(history)[-self.max_history :]
            if item in self.row_by_item
        ]
        if not rows:
            return None
        profile = sparse.csr_matrix(self.item_matrix[rows].mean(axis=0))
        return normalize(profile, norm="l2", copy=False)

    def recommend(self, history: list[int], k: int) -> list[Candidate]:
        """Return deterministic cosine-similarity recommendations, excluding seen items."""
        if k < 1:
            return []
        profile = self._profile(history)
        if profile is None or self.item_matrix is None:
            return []
        scores = (profile @ self.item_matrix.T).toarray().ravel()
        for item in set(history):
            row = self.row_by_item.get(int(item))
            if row is not None:
                scores[row] = -np.inf
        valid = np.isfinite(scores) & (scores > 0)
        valid_rows = np.flatnonzero(valid)
        if len(valid_rows) > k:
            selected = valid_rows[np.argpartition(-scores[valid_rows], k - 1)[:k]]
        else:
            selected = valid_rows
        selected = sorted(selected, key=lambda row: (-scores[row], int(self.item_ids[row])))
        return [
            Candidate(int(self.item_ids[row]), float(scores[row]), "content")
            for row in selected[:k]
        ]

    def recommend_batch(
        self,
        histories: list[list[int]],
        k: int,
        *,
        batch_size: int = 256,
    ) -> list[list[int]]:
        """Retrieve many users with bounded dense score batches."""
        if self.item_matrix is None:
            raise RuntimeError("TfidfContentRecommender must be fitted before recommendation")
        output: list[list[int]] = []
        empty = sparse.csr_matrix((1, self.item_matrix.shape[1]), dtype=np.float64)
        for start in range(0, len(histories), batch_size):
            batch = histories[start : start + batch_size]
            batch_profiles = []
            for history in batch:
                profile = self._profile(history)
                batch_profiles.append(profile if profile is not None else empty)
            profiles = sparse.vstack(batch_profiles)
            scores = (profiles @ self.item_matrix.T).toarray()
            for row, history in enumerate(batch):
                for item in set(history):
                    position = self.row_by_item.get(int(item))
                    if position is not None:
                        scores[row, position] = -np.inf
                valid = np.flatnonzero(np.isfinite(scores[row]) & (scores[row] > 0))
                if len(valid) > k:
                    selected = valid[np.argpartition(-scores[row, valid], k - 1)[:k]]
                else:
                    selected = valid
                selected = sorted(
                    selected,
                    key=lambda position: (
                        -scores[row, position],
                        int(self.item_ids[position]),
                    ),
                )
                output.append([int(self.item_ids[position]) for position in selected[:k]])
        return output
