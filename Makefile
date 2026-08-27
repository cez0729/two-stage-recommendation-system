.PHONY: install lint test smoke download clean split baselines faiss ranker ranker-test api feedback-export retrain-dry-run demo

install:
	python -m pip install -e ".[dev]"

lint:
	python -m ruff check .

test:
	python -m pytest

smoke:
	python -m pytest tests/test_import.py

download:
	python -m recsys.data.download --config configs/data.yaml

clean:
	python -m recsys.data.clean --config configs/data.yaml

split:
	python -m recsys.data.split --config configs/data.yaml

baselines:
	python -m recsys.retrieval.popularity --config configs/popularity.yaml
	python -m recsys.retrieval.itemcf --config configs/itemcf.yaml
	python -m recsys.evaluation.compare_baselines --results-dir results/video_games_2018/baselines

faiss:
	python -m recsys.retrieval.faiss_index --config configs/faiss.yaml

ranker:
	python -m recsys.ranking.pipeline --config configs/ranker.yaml

ranker-test:
	python -m recsys.ranking.evaluate --config configs/ranker.yaml

api:
	python -m uvicorn recsys.serving.api:app --host 127.0.0.1 --port 8000

feedback-export:
	python scripts/export_feedback.py --output runs/feedback.jsonl

retrain-dry-run:
	python scripts/retrain.py --config configs/retrain.yaml --dry-run

demo:
	@echo "Streamlit demo will be added in a later work package."
