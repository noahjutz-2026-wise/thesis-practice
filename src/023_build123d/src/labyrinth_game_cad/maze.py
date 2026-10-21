import json

from build123d import *

WIDTH = 290
LENGTH = 245
HEIGHT = 22

OUTER_WALL_WIDTH = 7.5
WALL_WIDTH = 5
WALL_HEIGHT = 6
HOLE_DIAMETER = 14.5


def maze(layout_path: str) -> BuildPart:
    with open(layout_path) as f:
        layout = json.load(f)
        with BuildPart() as part:
            with BuildSketch(Plane.XY) as plate:
                Rectangle(WIDTH, LENGTH)
            extrude(amount=-2)
            topf = part.faces().sort_by(Axis.Z).last
            with BuildSketch(topf) as walls:
                Rectangle(WIDTH, LENGTH)
                Rectangle(
                    WIDTH - 2 * OUTER_WALL_WIDTH,
                    LENGTH - 2 * OUTER_WALL_WIDTH,
                    mode=Mode.SUBTRACT,
                )
            extrude(amount=HEIGHT)
            with BuildSketch(Plane.XY) as holes:
                for x, z in layout["holes"]:
                    x = x * M - WIDTH / 2 + OUTER_WALL_WIDTH
                    z = z * M - LENGTH / 2 + OUTER_WALL_WIDTH
                    with Locations((x, z)):
                        Circle(HOLE_DIAMETER / 2)
            extrude(until=Until.FIRST, mode=Mode.SUBTRACT)

        return part
