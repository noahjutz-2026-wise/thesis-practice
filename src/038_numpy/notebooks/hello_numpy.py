import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Progress Grid Lookup Table
    """)
    return


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
    w = np.concat([wh, wv])
    w.shape
    return


@app.cell
def _():
    # Maze Layout
    X = 290
    Y = 245
    return X, Y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Meshgrid
    """)
    return


@app.cell
def _(X, Y, np):
    # Grid
    x = np.linspace(-X/2, X/2, 50)
    y = np.linspace(-Y/2, Y/2, 50)

    xv, yv = np.meshgrid(x, y)
    return xv, yv


@app.cell
def _(pd, xv, yv):
    z = xv ** 2 + yv ** 2
    df = pd.DataFrame({'x': xv.ravel(), 'y': yv.ravel(), 'z': z.ravel()})
    df
    return (df,)


@app.cell
def _(alt, df, mo):
    chart = alt.Chart(df).mark_rect().encode(
        x=alt.X('x:O').axis(values=df['x'][::5]),
        y=alt.Y('y:O').axis(values=df['y'][::500]),
        color='z'
    ).properties(
        width=300,
        height=300
    )
    mo.ui.altair_chart(chart)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Custom Matrix
    """)
    return


@app.cell
def _(X, Y, np):
    cell = 0.001 # meters
    origin = np.array([-X/2, -Y/2])
    origin
    return


if __name__ == "__main__":
    app.run()
