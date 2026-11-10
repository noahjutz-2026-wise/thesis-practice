import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path

from build123d import export_stl

from labyrinth_game_cad.maze import maze


def get_coacd_executable() -> str:
    # Resolve the CLI tool from the current virtualenv or PATH
    venv_bin = Path(sys.executable).parent / "coacd"
    if venv_bin.is_file():
        return str(venv_bin)
    coacd_bin = shutil.which("coacd")
    if coacd_bin:
        return coacd_bin
    raise FileNotFoundError(
        "Could not find 'coacd' executable in virtual environment or PATH."
    )


def main():
    f = (
        resources.files("labyrinth_game_cad")
        / "layouts"
        / "cyberrunner_hard_layout.json"
    )
    floor, outer, walls = maze(f)
    components = [
        ("floor", floor.part),
        ("outer", outer.part),
        ("walls", walls.part),
    ]

    coacd_bin = get_coacd_executable()

    for name, part in components:
        input_stl = Path(f"{name}_raw.stl")
        output_stl = Path(f"{name}.stl")

        export_stl(part, str(input_stl))
        print(f"Exported raw {name} mesh to {input_stl}")

        cmd = [
            coacd_bin,
            "-i",
            str(input_stl),
            "-o",
            str(output_stl),
            "-t",
            "0.005",
            "--split",
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
