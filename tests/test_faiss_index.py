import numpy as np

from recsys.retrieval.faiss_index import FaissIndexFlatIP


def test_faiss_matches_numpy_and_filters_seen_items(tmp_path) -> None:
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(20, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    item_ids = np.arange(100, 120)
    queries = vectors[[2, 7]]
    retriever = FaissIndexFlatIP.build(vectors, item_ids)

    actual_ids, _ = retriever.search(queries, k=5)
    expected_rows = np.argsort(-(queries @ vectors.T), axis=1)[:, :5]
    assert np.array_equal(actual_ids, item_ids[expected_rows])

    filtered_ids, _ = retriever.search(queries[:1], k=5, seen_item_ids=[{102, 103}])
    assert 102 not in filtered_ids[0]
    assert 103 not in filtered_ids[0]

    index_path = tmp_path / "items.index"
    ids_path = tmp_path / "ids.npy"
    retriever.save(index_path, ids_path)
    loaded = FaissIndexFlatIP.load(index_path, ids_path)
    loaded_ids, _ = loaded.search(queries, k=5)
    assert np.array_equal(loaded_ids, actual_ids)
