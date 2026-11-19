import json
from pathlib import Path

LAYOUT_PATH = (
    Path(__file__).parents[3]
    / "706_maze_build123d/src/labyrinth_game_cad/layouts/cyberrunner_hard_layout.json"
)


def main():
    with open(LAYOUT_PATH, "r") as f:
        layout = json.load(f)

    print(layout.keys())
