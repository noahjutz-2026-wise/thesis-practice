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
    import math

    return Path, alt, json, math, mo, np, pd


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
    ## Meshgrid (Linspace)
    """)
    return


@app.cell
def _(X, Y, alt, mo, np, pd):
    _x = np.linspace(-X/2, X/2, 50)
    _y = np.linspace(-Y/2, Y/2, 50)

    _xv, _yv = np.meshgrid(_x, _y)
    _z = _xv ** 2 + _yv ** 2
    _df = pd.DataFrame({'x': _xv.ravel(), 'y': _yv.ravel(), 'z': _z.ravel()})

    _chart = alt.Chart(_df).mark_rect().encode(
        x=alt.X('x:O').axis(values=_df['x'][::5]),
        y=alt.Y('y:O').axis(values=_df['y'][::500]),
        color='z'
    ).properties(    width=300,
        height=300
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Custom Matrix
    """)
    return


@app.cell
def _(X, Y, math, np):
    cell = 1 # mm
    origin = np.array([-X/2, -Y/2])
    shape = (math.ceil(X / cell), math.ceil(Y / cell))

    xs = origin[0] + (np.arange(shape[0]) + 0.5) * cell
    ys = origin[1] + (np.arange(shape[1]) + 0.5) * cell
    Xg, Yg = np.meshgrid(xs, ys, indexing='ij')
    return Xg, Yg


@app.cell
def _(Xg, Yg, alt, mo, pd):
    _z = Xg
    _df = pd.DataFrame({'x': Xg.ravel(), 'y': Yg.ravel(), 'z': _z.ravel()})

    _chart = alt.Chart(_df).mark_rect().encode(
        x=alt.X('x:O').axis(values=_df['x'][::5]),
        y=alt.Y('y:O').axis(values=_df['y'][::500]),
        color='z'
    ).properties(
        width=300,
        height=300
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Value Function

    $(x, y) \to p$
    """)
    return


if __name__ == "__main__":
    app.run()
