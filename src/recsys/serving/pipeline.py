"""Single-load online recommendation pipeline built from frozen artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from pathlib import Path
from time import perf_counter
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from recsys.config import load_yaml
from recsys.ranking.pipeline import FEATURE_COLUMNS, _reciprocal_rank, fuse_candidates
from recsys.retrieval.content import TfidfContentRecommender
from recsys.retrieval.faiss_index import FaissIndexFlatIP
from recsys.retrieval.itemcf import ItemCFRecommender
from recsys.retrieval.popularity import PopularityRecommender
from recsys.retrieval.two_tower import TwoTowerModel
from recsys.serving.versioning import resolve_serving_config


@dataclass(frozen=True)
class Recommendation:
    item_idx: int
    item_id: str
    title: str
    score: float
    source_count: int


class ServingPipeline:
    """Load all frozen artifacts once and serve deterministic recommendations."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        interactions_path = config["data"]["interactions_path"]
        interactions = pd.read_parquet(interactions_path)
        self.interactions = interactions
        serving = config["serving"]
        self.model_version = str(serving.get("version", "v1"))
        allowed_splits = set(
            serving.get("history_splits", ["retrieval_train", "rank_train", "validation"])
        )
        history = interactions[interactions["split"].isin(allowed_splits)].sort_values(
            ["user_idx", "timestamp", "item_idx"], kind="stable"
        )
        self.histories = {
            str(user_id): group.copy()
            for user_id, group in history.groupby("user_id", sort=False)
        }
        self.user_idx_by_id = {
            str(user_id): int(user_idx)
            for user_id, user_idx in interactions[["user_id", "user_idx"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        }
        self.item_rows = pd.read_parquet(config["data"]["items_path"]).set_index("item_idx")
        content_config = serving.get("content", {})
        self.content: TfidfContentRecommender | None = None
        if bool(content_config.get("enabled", False)):
            content_items = pd.read_parquet(content_config["items_path"])
            self.content = TfidfContentRecommender.load_artifacts(
                vectorizer_path=content_config["vectorizer_path"],
                item_matrix_path=content_config["item_matrix_path"],
                item_ids_path=content_config["item_ids_path"],
                items=content_items,
                max_history=int(content_config.get("max_history", 50)),
            )
        self.retrieval_train = interactions[interactions["split"] == "retrieval_train"]
        self.popularity = PopularityRecommender().fit(self.retrieval_train)
        itemcf_config = serving["itemcf"]
        self.itemcf = ItemCFRecommender(
            max_history=int(itemcf_config["max_history"]),
            top_neighbors=int(itemcf_config["top_neighbors"]),
            recency_decay=float(itemcf_config["recency_decay"]),
        ).fit(self.retrieval_train)
        self.faiss = FaissIndexFlatIP.load(
            serving["faiss_index_path"], serving["faiss_item_ids_path"]
        )
        checkpoint = torch.load(
            serving["two_tower_checkpoint"], map_location="cpu", weights_only=False
        )
        self.two_tower = TwoTowerModel(**checkpoint["model_args"])
        self.two_tower.load_state_dict(checkpoint["model_state_dict"])
        self.two_tower.eval()
        self.max_history = int(checkpoint["model_args"]["max_history"])
        self.ranker = lgb.Booster(model_file=serving["ranker_path"])
        self.catalog = set(self.faiss.item_ids.astype(int).tolist())
        self.popularity_rank = {
            candidate.item_idx: rank
            for rank, candidate in enumerate(self.popularity.ranking, start=1)
        }
        self.popularity_score = {
            item: log1p(count) for item, count in self.popularity.item_counts.items()
        }
        self.item_stats = self.retrieval_train.groupby("item_idx").agg(
            item_mean_rating=("rating", "mean"),
            item_rating_std=("rating", lambda values: values.std(ddof=0)),
        )
        self.rrf_constant = int(serving.get("rrf_constant", 60))
        self.pool_size = int(serving.get("pool_size", 200))
        self.two_tower_candidates = int(serving.get("two_tower_candidates", 100))
        self.itemcf_candidates = int(serving.get("itemcf_candidates", 100))
        self.popularity_candidates = int(serving.get("popularity_candidates", 20))
        self.content_candidates = int(serving.get("content_candidates", 100))

    @classmethod
    def from_yaml(cls, path: str | Path) -> ServingPipeline:
        return cls(resolve_serving_config(load_yaml(path), Path(path).resolve().parents[1]))

    @property
    def catalog_size(self) -> int:
        return len(self.catalog)

    def _history_tensor(self, history: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        recent = history[-self.max_history :]
        padded = [item + 1 for item in recent]
        padded.extend([0] * (self.max_history - len(padded)))
        return torch.tensor([padded], dtype=torch.long), torch.tensor(
            [len(recent)], dtype=torch.long
        )

    def _candidate_features(
        self,
        history_frame: pd.DataFrame,
        fused: list[dict[str, float | int]],
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        frame = pd.DataFrame(fused)
        candidate_ids = frame["item_idx"].astype(int).to_numpy()
        defaults = {
            "two_tower_score": -2.0,
            "two_tower_rank": self.two_tower_candidates + 1,
            "content_score": 0.0,
            "content_rank": self.content_candidates + 1,
            "itemcf_score": 0.0,
            "itemcf_rank": self.itemcf_candidates + 1,
        }
        for column, default in defaults.items():
            if column not in frame:
                frame[column] = default
            else:
                frame[column] = frame[column].fillna(default)
        frame["popularity_score"] = frame["item_idx"].map(self.popularity_score).fillna(0.0)
        frame["popularity_rank"] = frame["item_idx"].map(self.popularity_rank).fillna(
            len(self.catalog) + 1
        )
        frame["user_history_length"] = len(history_frame)
        frame["user_mean_rating"] = (
            float(history_frame["rating"].mean()) if len(history_frame) else 0.0
        )
        frame["user_rating_std"] = (
            float(history_frame["rating"].std(ddof=0)) if len(history_frame) else 0.0
        )
        if len(history_frame):
            last_timestamp = pd.Timestamp(history_frame["timestamp"].max())
            reference_timestamp = as_of or pd.Timestamp("2018-01-01", tz="UTC")
            frame["days_since_last_action"] = max(
                0.0, (reference_timestamp - last_timestamp).total_seconds() / 86400
            )
        else:
            frame["days_since_last_action"] = 0.0
        frame["item_popularity_log"] = frame["popularity_score"]
        stats = self.item_stats.reindex(candidate_ids)
        frame["item_mean_rating"] = stats["item_mean_rating"].fillna(0.0).to_numpy()
        frame["item_rating_std"] = stats["item_rating_std"].fillna(0.0).to_numpy()
        if frame[FEATURE_COLUMNS].isna().any().any():
            raise ValueError("Serving feature vector contains missing values")
        return frame

    def _popularity_fallback(self, history: set[int], k: int) -> list[Recommendation]:
        output = []
        for candidate in self.popularity.recommend(history, k=k):
            row = (
                self.item_rows.loc[int(candidate.item_idx)]
                if int(candidate.item_idx) in self.item_rows.index
                else {}
            )
            output.append(
                Recommendation(
                    item_idx=int(candidate.item_idx),
                    item_id=str(row.get("item_id", candidate.item_idx)),
                    title=str(row.get("title", "")),
                    score=float(candidate.score),
                    source_count=1,
                )
            )
        return output

    def recommend(
        self, user_id: str, k: int = 10, as_of: pd.Timestamp | None = None
    ) -> list[Recommendation]:
        """Return deterministic top-k recommendations for a known or unknown user."""
        recommendations, _ = self.recommend_with_trace(user_id, k, as_of=as_of)
        return recommendations

    def recommend_with_trace(
        self, user_id: str, k: int = 10, as_of: pd.Timestamp | None = None
    ) -> tuple[list[Recommendation], dict[str, float | int | bool]]:
        """Return recommendations plus per-stage latency without changing ranking behavior."""
        total_started = perf_counter()
        stage_latency_ms: dict[str, float] = {}
        if not 1 <= k <= 50:
            raise ValueError("k must be between 1 and 50")

        stage_started = perf_counter()
        history_frame = self.histories.get(str(user_id))
        stage_latency_ms["history_prepare"] = (perf_counter() - stage_started) * 1000
        if history_frame is None or history_frame.empty:
            stage_started = perf_counter()
            output = self._popularity_fallback(set(), k)
            stage_latency_ms["fallback_popularity"] = (perf_counter() - stage_started) * 1000
            trace: dict[str, float | int | bool] = {
                **stage_latency_ms,
                "candidate_count": len(output),
                "fallback": True,
                "total_pipeline_ms": (perf_counter() - total_started) * 1000,
            }
            return output, trace

        stage_started = perf_counter()
        history = history_frame["item_idx"].astype(int).tolist()
        seen = set(history)
        user_idx = self.user_idx_by_id[str(user_id)]
        history_tensor, history_lengths = self._history_tensor(history)
        stage_latency_ms["history_prepare"] += (perf_counter() - stage_started) * 1000

        stage_started = perf_counter()
        with torch.inference_mode():
            user_vector = self.two_tower.encode_users(
                torch.tensor([user_idx], dtype=torch.long), history_tensor, history_lengths
            ).numpy()
        stage_latency_ms["two_tower_encode"] = (perf_counter() - stage_started) * 1000

        stage_started = perf_counter()
        two_ids, two_scores = self.faiss.search(
            user_vector, k=self.two_tower_candidates, seen_item_ids=[seen]
        )
        stage_latency_ms["faiss_search"] = (perf_counter() - stage_started) * 1000

        stage_started = perf_counter()
        itemcf_candidates = [
            (candidate.item_idx, candidate.score)
            for candidate in self.itemcf.recommend(history, k=self.itemcf_candidates)
        ]
        stage_latency_ms["itemcf"] = (perf_counter() - stage_started) * 1000

        stage_started = perf_counter()
        popularity_candidates = [
            (candidate.item_idx, candidate.score)
            for candidate in self.popularity.recommend(history, k=self.popularity_candidates)
        ]
        stage_latency_ms["popularity"] = (perf_counter() - stage_started) * 1000

        stage_started = perf_counter()
        content_candidates = []
        if self.content is not None:
            content_candidates = [
                (candidate.item_idx, candidate.score)
                for candidate in self.content.recommend(history, k=self.content_candidates)
            ]
        stage_latency_ms["content"] = (perf_counter() - stage_started) * 1000

        stage_started = perf_counter()
        fused = fuse_candidates(
            list(zip(two_ids[0].tolist(), two_scores[0].tolist(), strict=True)),
            itemcf_candidates,
            popularity_candidates,
            content_candidates if self.content is not None else None,
            pool_size=self.pool_size,
            rrf_constant=self.rrf_constant,
        )
        if len(fused) < self.pool_size:
            emitted = {int(row["item_idx"]) for row in fused}
            for candidate in self.popularity.recommend(history, k=self.pool_size):
                if candidate.item_idx in emitted:
                    continue
                fused.append(
                    {
                        "item_idx": candidate.item_idx,
                        "popularity_score": candidate.score,
                        "popularity_rank": self.popularity_rank[candidate.item_idx],
                        "source_count": 1,
                        "rrf_score": _reciprocal_rank(
                            self.popularity_rank[candidate.item_idx], self.rrf_constant
                        ),
                        "best_source_rank": self.popularity_rank[candidate.item_idx],
                    }
                )
                emitted.add(candidate.item_idx)
                if len(fused) == self.pool_size:
                    break
        stage_latency_ms["candidate_merge"] = (perf_counter() - stage_started) * 1000

        candidate_count = len(fused)
        stage_started = perf_counter()
        features = self._candidate_features(history_frame, fused, as_of=as_of)
        stage_latency_ms["feature_build"] = (perf_counter() - stage_started) * 1000

        stage_started = perf_counter()
        predictions = self.ranker.predict(features[FEATURE_COLUMNS])
        order = np.argsort(-predictions, kind="stable")
        stage_latency_ms["ranker_predict"] = (perf_counter() - stage_started) * 1000

        stage_started = perf_counter()
        output = []
        emitted: set[int] = set()
        for position in order:
            item_idx = int(features.iloc[position]["item_idx"])
            if item_idx in seen or item_idx in emitted:
                continue
            emitted.add(item_idx)
            row = self.item_rows.loc[item_idx] if item_idx in self.item_rows.index else {}
            output.append(
                Recommendation(
                    item_idx=item_idx,
                    item_id=str(row.get("item_id", item_idx)),
                    title=str(row.get("title", "")),
                    score=float(predictions[position]),
                    source_count=int(features.iloc[position]["source_count"]),
                )
            )
            if len(output) == k:
                break
        if len(output) < k:
            fallback = self._popularity_fallback(
                seen | {item.item_idx for item in output}, k - len(output)
            )
            output.extend(fallback)
        output = output[:k]
        stage_latency_ms["result_assembly"] = (perf_counter() - stage_started) * 1000
        trace = {
            **stage_latency_ms,
            "candidate_count": candidate_count,
            "fallback": False,
            "total_pipeline_ms": (perf_counter() - total_started) * 1000,
        }
        return output, trace

    def recommend_at(
        self, user_id: str, k: int = 10, as_of: pd.Timestamp | None = None
    ) -> list[Recommendation]:
        """Explicit cutoff-aware variant used only for offline parity checks."""
        return self.recommend(user_id, k, as_of=as_of)
