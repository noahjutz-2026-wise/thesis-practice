import sys
import time
from importlib import resources

from yacv_server import show

from labyrinth_game_cad.maze import maze


def main():
    f = (
        resources.files("labyrinth_game_cad")
        / "layouts"
        / "cyberrunner_hard_layout.json"
    )
    show(*maze(f))
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            sys.exit()


if __name__ == "__main__":
    main()
