import json

from build123d import *

WIDTH = 290
LENGTH = 245
HEIGHT = 22

OUTER_WALL_WIDTH = 7.5
WALL_WIDTH = 5
WALL_HEIGHT = 6
HOLE_DIAMETER = 14.5


def maze(layout_path: str) -> tuple[BuildPart, BuildPart, BuildPart]:
    with open(layout_path) as f:
        layout = json.load(f)
        with BuildPart() as floor:
            with BuildSketch(Plane.XY) as _plate:
                Rectangle(WIDTH, LENGTH)
            extrude(amount=-2)

            with BuildSketch(Plane.XY) as _holes:
                for y, z in layout["holes"]:
                    y = y * M - WIDTH / 2 + OUTER_WALL_WIDTH
                    z = z * M - LENGTH / 2 + OUTER_WALL_WIDTH
                    with Locations((y, z)):
                        Circle(HOLE_DIAMETER / 2)
            extrude(until=Until.FIRST, mode=Mode.SUBTRACT)

        with BuildPart() as outer_walls:
            with BuildSketch(Plane.XY) as _outer_walls:
                Rectangle(WIDTH, LENGTH)
                Rectangle(
                    WIDTH - 2 * OUTER_WALL_WIDTH,
                    LENGTH - 2 * OUTER_WALL_WIDTH,
                    mode=Mode.SUBTRACT,
                )
            extrude(amount=HEIGHT)

        with BuildPart() as walls:
            with BuildSketch(Plane.XY) as _walls:
                for a, b, y in layout["walls_v"]:
                    a = a * M - LENGTH / 2 + OUTER_WALL_WIDTH
                    b = b * M - LENGTH / 2 + OUTER_WALL_WIDTH
                    y = y * M - WIDTH / 2 + OUTER_WALL_WIDTH
                    with BuildLine() as ln:
                        Line((y, a), (y, b))
                    trace(ln.line, line_width=WALL_WIDTH)
                for a, b, y in layout["walls_h"]:
                    a = a * M - WIDTH / 2 + OUTER_WALL_WIDTH
                    b = b * M - WIDTH / 2 + OUTER_WALL_WIDTH
                    y = y * M - LENGTH / 2 + OUTER_WALL_WIDTH
                    with BuildLine() as ln:
                        Line((a, y), (b, y))
                    trace(ln.line, line_width=WALL_WIDTH)
            extrude(amount=WALL_HEIGHT)

        return floor, outer_walls, walls
