from build123d import *


def main():
    with BuildPart() as part:
        with BuildSketch() as base_sketch:
            Rectangle(20, 20)
        extrude(amount=10)
        with BuildSketch(part.faces().sort_by(Axis.Z).last) as cut_sketch:
            Circle(5)
        extrude(amount=-5, mode=Mode.SUBTRACT)

    result = part.part

    export_step(result, "out.step")
