import time

from yacv_server import show

from labyrinth_game_cad.maze import maze


def main():
    show([maze()])
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
