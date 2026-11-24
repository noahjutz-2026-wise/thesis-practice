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
    import cv2

    return Path, alt, json, math, mo, np, pd


@app.cell
def _(Path, json):
    LAYOUT_XML = Path(__file__).parents[2] / "706_maze_build123d/src/labyrinth_game_cad/layouts/cyberrunner_hard_layout.json"
    with open(LAYOUT_XML, "r") as _f:
        layout = json.load(_f)
    return (layout,)


@app.cell
def _(layout, np):
    # Wall points (N, 2, 2)
    wh = np.array(layout["walls_h"])
    wv = np.array(layout["walls_v"])

    wh = wh[:, [[0, 2], [1, 2]]]
    wv = wv[:, [[2, 0], [2, 1]]]
    w = np.concat([wh, wv])
    # 0: N
    # 1: 2 (start/end)
    # 2: 2 (x/y)

    # Waypoints (N, 2)
    wp = np.array(layout["waypoints"])
    wp.shape
    return w, wp


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
    _z = _xv
    _df = pd.DataFrame({'x': _xv.ravel(), 'y': _yv.ravel(), 'z': _z.ravel()})

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


@app.cell
def _(np, w, wp, xv, yv):
    def f(x, y, O, wp):
        P = np.stack([x, y], axis=-1)
        A = wp[:-1]
        B = wp[1:]
        AB = B - A
        return AB.shape

    f(xv, yv, w, wp)
    return


@app.cell
def _(alt, mo, pd, wp):
    _df = pd.DataFrame(wp, columns=["x", "y"])
    _chart = alt.Chart(_df).mark_circle().encode(
        x='x', y='y'
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
    return cell, origin, shape


@app.cell
def _(alt, cell, mo, np, origin, pd, shape):
    _x = origin[0] + (np.arange(shape[0]) + 0.5) * cell
    _y = origin[1] + (np.arange(shape[1]) + 0.5) * cell
    xv, yv = np.meshgrid(_x, _y, indexing='ij')

    _z = xv
    _df = pd.DataFrame({'x': xv.ravel(), 'y': yv.ravel(), 'z': _z.ravel()})

    _chart = alt.Chart(_df).mark_rect().encode(
        x=alt.X('x:O').axis(values=_df['x'][::5000]),
        y=alt.Y('y:O').axis(values=_df['y'][::5000]),
        color='z'
    ).properties(
        width=300,
        height=300
    )
    mo.ui.altair_chart(_chart)
    return xv, yv


if __name__ == "__main__":
    app.run()
