from recsys.serving.api import create_app
from recsys.serving.pipeline import ServingPipeline


def test_serving_rejects_invalid_k() -> None:
    pipeline = object.__new__(ServingPipeline)
    try:
        pipeline.recommend("unknown", 0)
    except ValueError as exc:
        assert "between 1 and 50" in str(exc)
    else:
        raise AssertionError("invalid k was accepted")


def test_serving_history_tensor_is_padded() -> None:
    pipeline = object.__new__(ServingPipeline)
    pipeline.max_history = 3
    history, lengths = pipeline._history_tensor([4, 5])
    assert history.tolist() == [[5, 6, 0]]
    assert lengths.tolist() == [2]


def test_unknown_user_trace_reports_fallback() -> None:
    pipeline = object.__new__(ServingPipeline)
    pipeline.histories = {}
    pipeline._popularity_fallback = lambda history, k: []
    recommendations, trace = pipeline.recommend_with_trace("unknown", 3)
    assert recommendations == []
    assert trace["fallback"] is True
    assert trace["candidate_count"] == 0
    assert trace["fallback_popularity"] >= 0
    assert trace["total_pipeline_ms"] >= trace["fallback_popularity"]


def test_api_health_with_injected_pipeline() -> None:
    from fastapi.testclient import TestClient

    class StubPipeline:
        catalog_size = 10

    pipeline = StubPipeline()
    client = TestClient(create_app(pipeline=pipeline))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_api_unknown_user_and_metrics_with_stub() -> None:
    from fastapi.testclient import TestClient

    class StubPipeline:
        catalog_size = 10

        def recommend(self, user_id: str, k: int):
            return []

    client = TestClient(create_app(pipeline=StubPipeline()))
    response = client.get("/recommend/unknown?k=3")
    assert response.status_code == 200
    assert response.json() == {"user_id": "unknown", "k": 3, "recommendations": []}
    metrics = client.get("/metrics").json()
    assert metrics["request_count"] == 1


def test_api_cache_hit_is_observable() -> None:
    from fastapi.testclient import TestClient

    class StubPipeline:
        catalog_size = 10

        def recommend(self, user_id: str, k: int):
            return []

    class FakeCache:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value
            return True

        def ping(self):
            return True

    client = TestClient(create_app(pipeline=StubPipeline(), cache=FakeCache()))
    client.get("/recommend/u?k=2")
    client.get("/recommend/u?k=2")
    metrics = client.get("/metrics").json()
    assert metrics["cache_hit"] == 1
    assert metrics["cache_miss"] == 1


def test_api_metadata_events_and_prometheus(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from recsys.serving.events import SQLiteFeedbackStore

    class StubPipeline:
        catalog_size = 10
        model_version = "test-v1"
        histories = {"known": object()}

        def recommend(self, user_id: str, k: int):
            return []

    store = SQLiteFeedbackStore(tmp_path / "events.db")
    client = TestClient(create_app(pipeline=StubPipeline(), feedback_store=store))
    recommendation = client.get("/recommend/known?k=2").json()
    assert recommendation["model_version"] == "test-v1"
    assert recommendation["request_id"]
    assert recommendation["fallback"] is False

    event = {
        "user_id": "known",
        "item_id": "item-1",
        "event_type": "click",
        "timestamp": "2026-08-27T12:00:00+00:00",
        "request_id": recommendation["request_id"],
        "model_version": recommendation["model_version"],
        "rank_position": 1,
        "simulated": True,
    }
    response = client.post("/events", json=event)
    assert response.status_code == 200
    assert response.json()["accepted"] is True
    prometheus = client.get("/metrics?format=prometheus")
    assert prometheus.status_code == 200
    assert "recsys_request_count 1" in prometheus.text
    assert 'recsys_model_version_info{version="test-v1"} 1' in prometheus.text

    exported = tmp_path / "events.jsonl"
    assert store.export(exported) == 1
    assert '"event_type": "click"' in exported.read_text(encoding="utf-8")


def test_api_exports_pipeline_stage_metrics() -> None:
    from fastapi.testclient import TestClient

    class TracedStubPipeline:
        catalog_size = 10
        model_version = "trace-v1"
        histories = {"known": object()}

        def recommend_with_trace(self, user_id: str, k: int):
            return [], {
                "history_prepare": 1.0,
                "faiss_search": 2.0,
                "candidate_count": 200,
                "fallback": False,
                "total_pipeline_ms": 3.0,
            }

    client = TestClient(create_app(pipeline=TracedStubPipeline()))
    response = client.get("/recommend/known?k=3")
    assert response.status_code == 200
    metrics = client.get("/metrics").json()
    assert metrics["stage_latency_ms"]["faiss_search"] == {"p50": 2.0, "p95": 2.0}
    assert metrics["candidate_count_latest"] == 200
    prometheus = client.get("/metrics?format=prometheus").text
    expected_metric = (
        'recsys_stage_latency_seconds{stage="faiss_search",quantile="0.95"} 0.002000'
    )
    assert expected_metric in prometheus
    assert "recsys_candidate_count 200" in prometheus
