import json
import os
from pathlib import Path

from dvc_data_prep.sources.get_wandb import episode_returns

JSON_PATH = Path(__file__).parents[3] / "outs" / "wandb_returns.json"


def main():
    os.makedirs(JSON_PATH.parent, exist_ok=True)
    with open(JSON_PATH, "w") as f:
        json.dump(episode_returns(), f, indent=2)


if __name__ == "__main__":
    main()
