"""Build, validate, persist, and benchmark an exact FAISS retrieval index."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch

from recsys.config import load_yaml
from recsys.retrieval.dataset import build_evaluation_queries
from recsys.retrieval.two_tower import TwoTowerModel
from recsys.utils.io import sha256_file, write_json
from recsys.utils.seed import seed_everything


class FaissIndexFlatIP:
    """Exact inner-product retrieval with stable external item identifiers."""

    def __init__(self, index: faiss.Index, item_ids: np.ndarray) -> None:
        self.index = index
        self.item_ids = np.asarray(item_ids, dtype=np.int64)
        if self.index.ntotal != len(self.item_ids):
            raise ValueError("FAISS rows and item_ids must have the same length")

    @classmethod
    def build(cls, item_vectors: np.ndarray, item_ids: np.ndarray) -> "FaissIndexFlatIP":
        vectors = np.ascontiguousarray(item_vectors, dtype=np.float32)
        ids = np.asarray(item_ids, dtype=np.int64)
        if vectors.ndim != 2 or vectors.shape[0] != len(ids):
            raise ValueError("item_vectors must be [num_items, dim] and align with item_ids")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("item_ids must be unique")
        if not np.isfinite(vectors).all():
            raise ValueError("item_vectors contain NaN or infinity")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("IndexFlatIP expects L2-normalized item vectors")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, ids)

    def search(
        self,
        query_vectors: np.ndarray,
        *,
        k: int,
        seen_item_ids: list[set[int]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve exact top-k external IDs, optionally excluding seen items."""
        queries = np.ascontiguousarray(query_vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]
        if queries.ndim != 2 or queries.shape[1] != self.index.d:
            raise ValueError("query_vectors have the wrong shape")
        if not np.isfinite(queries).all():
            raise ValueError("query_vectors contain NaN or infinity")
        if k <= 0 or k > self.index.ntotal:
            raise ValueError("k must be between 1 and the index size")
        if seen_item_ids is None:
            seen_item_ids = [set() for _ in range(len(queries))]
        if len(seen_item_ids) != len(queries):
            raise ValueError("seen_item_ids must align with query_vectors")

        max_seen = max((len(seen) for seen in seen_item_ids), default=0)
        search_k = min(self.index.ntotal, k + max_seen)
        raw_scores, raw_rows = self.index.search(queries, search_k)
        result_ids = np.full((len(queries), k), -1, dtype=np.int64)
        result_scores = np.full((len(queries), k), -np.inf, dtype=np.float32)
        for row in range(len(queries)):
            output_position = 0
            seen = seen_item_ids[row]
            for score, internal_id in zip(raw_scores[row], raw_rows[row], strict=True):
                item_id = int(self.item_ids[internal_id])
                if item_id in seen:
                    continue
                result_ids[row, output_position] = item_id
                result_scores[row, output_position] = score
                output_position += 1
                if output_position == k:
                    break
            if output_position < k:
                raise RuntimeError("Not enough unseen items to satisfy requested k")
        return result_ids, result_scores

    def save(self, index_path: str | Path, item_ids_path: str | Path) -> None:
        index_path = Path(index_path)
        item_ids_path = Path(item_ids_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        item_ids_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        np.save(item_ids_path, self.item_ids)

    @classmethod
    def load(cls, index_path: str | Path, item_ids_path: str | Path) -> "FaissIndexFlatIP":
        return cls(faiss.read_index(str(index_path)), np.load(item_ids_path))


def _load_user_vectors(config: dict[str, Any]) -> tuple[np.ndarray, list[set[int]]]:
    checkpoint = torch.load(
        config["model"]["checkpoint_path"], map_location="cpu", weights_only=False
    )
    model = TwoTowerModel(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    interactions = pd.read_parquet(config["data"]["interactions_path"])
    catalog = set(np.load(config["data"]["item_ids_path"]).astype(int).tolist())
    queries = build_evaluation_queries(
        interactions,
        split="validation",
        training_catalog=catalog,
        max_history=int(checkpoint["model_args"]["max_history"]),
    )
    with torch.inference_mode():
        vectors = model.encode_users(
            queries.user_ids,
            queries.histories,
            queries.history_lengths,
        ).numpy()
    return vectors, queries.seen_histories


def build_and_benchmark(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    seed_everything(int(config["seed"]))
    vectors = np.load(config["data"]["item_vectors_path"])
    item_ids = np.load(config["data"]["item_ids_path"])
    started = perf_counter()
    retriever = FaissIndexFlatIP.build(vectors, item_ids)
    build_seconds = perf_counter() - started
    retriever.save(config["outputs"]["index_path"], config["outputs"]["item_ids_path"])

    user_vectors, seen_histories = _load_user_vectors(config)
    rng = np.random.default_rng(int(config["seed"]))
    sample_size = min(int(config["benchmark"]["correctness_users"]), len(user_vectors))
    sample_rows = rng.choice(len(user_vectors), size=sample_size, replace=False)
    sample_vectors = user_vectors[sample_rows]
    k = int(config["benchmark"]["k"])
    faiss_ids, _ = retriever.search(sample_vectors, k=k)
    numpy_scores = sample_vectors @ vectors.T
    numpy_rows = np.argsort(-numpy_scores, axis=1, kind="stable")[:, :k]
    numpy_ids = item_ids[numpy_rows]
    exact_row_match_rate = float(np.mean(np.all(faiss_ids == numpy_ids, axis=1)))
    mean_overlap = float(
        np.mean(
            [
                len(set(left) & set(right)) / k
                for left, right in zip(faiss_ids, numpy_ids, strict=True)
            ]
        )
    )

    single_latencies = []
    single_vector = user_vectors[:1]
    single_seen = seen_histories[:1]
    for _ in range(int(config["benchmark"]["single_repetitions"])):
        started = perf_counter()
        retriever.search(single_vector, k=k, seen_item_ids=single_seen)
        single_latencies.append((perf_counter() - started) * 1000)
    batch_size = min(int(config["benchmark"]["batch_size"]), len(user_vectors))
    batch_latencies = []
    for _ in range(int(config["benchmark"]["batch_repetitions"])):
        started = perf_counter()
        retriever.search(
            user_vectors[:batch_size],
            k=k,
            seen_item_ids=seen_histories[:batch_size],
        )
        batch_latencies.append((perf_counter() - started) * 1000)

    loaded = FaissIndexFlatIP.load(
        config["outputs"]["index_path"], config["outputs"]["item_ids_path"]
    )
    reload_ids, _ = loaded.search(sample_vectors[:1], k=k)
    payload = {
        "index_type": "IndexFlatIP",
        "metric": "inner_product",
        "num_items": int(retriever.index.ntotal),
        "embedding_dim": int(retriever.index.d),
        "build_seconds": build_seconds,
        "index_size_bytes": Path(config["outputs"]["index_path"]).stat().st_size,
        "correctness": {
            "users": sample_size,
            "k": k,
            "exact_row_match_rate": exact_row_match_rate,
            "mean_topk_overlap": mean_overlap,
            "reload_identical": bool(np.array_equal(reload_ids, faiss_ids[:1])),
        },
        "latency_ms": {
            "single_p50": float(np.percentile(single_latencies, 50)),
            "single_p95": float(np.percentile(single_latencies, 95)),
            "batch_size": batch_size,
            "batch_p50": float(np.percentile(batch_latencies, 50)),
            "batch_p95": float(np.percentile(batch_latencies, 95)),
            "batch_per_user_p95": float(np.percentile(batch_latencies, 95) / batch_size),
        },
        "artifacts": {
            "index_path": str(config["outputs"]["index_path"]),
            "item_ids_path": str(config["outputs"]["item_ids_path"]),
            "checkpoint_sha256": sha256_file(config["model"]["checkpoint_path"]),
            "vectors_sha256": sha256_file(config["data"]["item_vectors_path"]),
        },
    }
    write_json(config["outputs"]["metadata_path"], payload)
    write_json(config["outputs"]["benchmark_path"], payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/faiss.yaml")
    args = parser.parse_args()
    print(json.dumps(build_and_benchmark(args.config), indent=2))


if __name__ == "__main__":
    main()
