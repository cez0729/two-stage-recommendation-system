from pathlib import Path

from recsys.config import load_yaml


def test_load_data_config() -> None:
    config = load_yaml(Path("configs/data.yaml"))

    assert config["project_seed"] == 42
    assert config["dataset"]["category"] == "Video_Games"
    assert config["dataset"]["review_schema"]["timestamp_unit"] == "s"
    assert config["paths"]["raw_dir"] == "data/raw/video_games_2018"
