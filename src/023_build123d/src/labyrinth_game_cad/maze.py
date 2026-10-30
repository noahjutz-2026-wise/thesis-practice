import json

from build123d import *

WIDTH = 290
LENGTH = 245
HEIGHT = 22

OUTER_WALL_WIDTH = 7.5
WALL_WIDTH = 5
WALL_HEIGHT = 6
HOLE_DIAMETER = 15.5


def maze(layout_path: str) -> tuple[BuildPart, BuildPart, BuildPart]:
    with open(layout_path) as f:
        layout = json.load(f)
        with BuildPart() as floor:
            with BuildSketch(Plane.XY) as _plate:
                Rectangle(WIDTH, LENGTH)
            extrude(amount=-2)

            with BuildSketch(Plane.XY) as _holes:
                for y, z in layout["holes"]:
                    with Locations((y * M, z * M)):
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
                    with BuildLine() as ln:
                        Line((y * M, a * M), (y * M, b * M))
                    trace(ln.line, line_width=WALL_WIDTH)
                for a, b, y in layout["walls_h"]:
                    with BuildLine() as ln:
                        Line((a * M, y * M), (b * M, y * M))
                    trace(ln.line, line_width=WALL_WIDTH)
            extrude(amount=WALL_HEIGHT)
            top_edges = walls.edges().filter_by(Plane.XY).group_by(Axis.Z)[-1]
            fillet(top_edges, radius=WALL_WIDTH / 2 * 0.85)

        return floor, outer_walls, walls
