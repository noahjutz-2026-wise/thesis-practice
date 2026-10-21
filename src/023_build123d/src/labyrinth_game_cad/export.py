from importlib import resources
from pathlib import Path

from build123d import export_stl

from labyrinth_game_cad.maze import maze


def main():
    f = (
        resources.files("labyrinth_game_cad")
        / "layouts"
        / "cyberrunner_hard_layout.json"
    )
    parts = maze(f)
    for i, part in enumerate(parts):
        part = part.part
        output_path = Path(f"part{i}.stl")
        export_stl(part, str(output_path))
        print(f"Exported maze to {output_path}")


if __name__ == "__main__":
    main()
