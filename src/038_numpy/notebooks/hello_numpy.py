import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from pathlib import Path
    import json
    import matplotlib.pyplot as plt
    import altair as alt
    import pandas as pd

    return Path, alt, json, mo, np, pd


@app.cell
def _(Path, json):
    LAYOUT_XML = Path(__file__).parents[2] / "706_maze_build123d/src/labyrinth_game_cad/layouts/cyberrunner_hard_layout.json"
    with open(LAYOUT_XML, "r") as f:
        layout = json.load(f)
    return (layout,)


@app.cell
def _(layout, np):
    # Wall points (N, 2, 2)
    wh = np.array(layout["walls_h"])
    wv = np.array(layout["walls_v"])

    wh = wh[:, [[0, 2], [1, 2]]]
    wv = wv[:, [[2, 0], [2, 1]]]
    return


@app.cell
def _(np):
    # Grid
    X = 290
    Y = 245

    x = np.linspace(-X/2, X/2, 50)
    y = np.linspace(-Y/2, Y/2, 50)

    xv, yv = np.meshgrid(x, y)
    return xv, yv


@app.cell
def _(alt, mo, pd, xv, yv):
    z = xv + yv
    df = pd.DataFrame({'x': xv.ravel(), 'y': yv.ravel(), 'z': z.ravel()})

    chart = alt.Chart(df).mark_rect().encode(
        x='x:O',
        y='y:O',
        color='z'
    )
    mo.ui.altair_chart(chart)
    return


if __name__ == "__main__":
    app.run()
