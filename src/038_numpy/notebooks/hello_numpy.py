import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Progress Grid Lookup Table
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inputs
    """)
    return


@app.cell
def _():
    import json
    import math
    from pathlib import Path

    import altair as alt
    import cv2
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return Path, alt, json, math, mo, np, pd


@app.cell
def _(Path, json):
    LAYOUT_XML = (
        Path(__file__).parents[2]
        / "706_maze_build123d/src/labyrinth_game_cad/layouts/cyberrunner_hard_layout.json"
    )
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
    return (wp,)


@app.cell
def _():
    # Maze Layout
    X = 290 / 1000  # mm to m
    Y = 245 / 1000  # mm to m
    return X, Y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Meshgrid (Linspace)
    """)
    return


@app.cell
def _(X, Y, alt, mo, np, pd):
    _x = np.linspace(-X / 2, X / 2, 50)
    _y = np.linspace(-Y / 2, Y / 2, 50)

    _xv, _yv = np.meshgrid(_x, _y)
    _z = _xv
    _df = pd.DataFrame({"x": _xv.ravel(), "y": _yv.ravel(), "z": _z.ravel()})

    _chart = (
        alt.Chart(_df)
        .mark_rect()
        .encode(
            x=alt.X("x:O").axis(values=_df["x"][::5]),
            y=alt.Y("y:O").axis(values=_df["y"][::500]),
            color="z",
        )
        .properties(width=300, height=300)
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpolated Waypoints
    """)
    return


@app.cell
def _(np, wp):
    # Interpolated Waypoints
    _M = 500
    _s = np.diff(wp, axis=0)
    _s = np.linalg.norm(_s, axis=-1)
    _s = np.cumsum(_s)
    _s = np.pad(_s, (1, 0))

    _t = np.linspace(0, _s[-1], _M)
    _x = np.interp(_t, _s, wp[:, 0])
    _y = np.interp(_t, _s, wp[:, 1])
    wp_interp = np.stack([_x, _y], axis=-1)
    return (wp_interp,)


@app.cell(hide_code=True)
def _(alt, mo, pd, wp_interp):
    _df = pd.DataFrame(wp_interp, columns=["x", "y"])
    _chart = alt.Chart(_df).mark_circle().encode(x="x", y="y")
    mo.ui.altair_chart(_chart)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2D Function
    """)
    return


@app.cell
def _(np):
    def f(xv, yv):
        P = np.stack([xv, yv], axis=-1)
        return yv

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Custom Matrix
    """)
    return


@app.cell
def _(X, Y, math, np):
    cell = 0.001  # m
    origin = np.array([-X / 2, -Y / 2])
    shape = (math.ceil(X / cell), math.ceil(Y / cell))
    return cell, origin, shape


@app.cell
def _(cell, np, origin, shape):
    _x = origin[0] + (np.arange(shape[0]) + 0.5) * cell
    _y = origin[1] + (np.arange(shape[1]) + 0.5) * cell
    xv, yv = np.meshgrid(_x, _y, indexing="ij")
    return xv, yv


@app.cell
def _(np, wp_interp, xv, yv):
    P = np.stack([xv, yv], axis=-1)
    _v = P[..., np.newaxis, :]
    _v = _v - wp_interp
    _d = np.linalg.norm(_v, axis=-1)

    z = _d.min(axis=-1)
    _v.shape
    return (z,)


@app.cell(hide_code=True)
def _(X, Y, alt, mo, pd, xv, yv, z):
    _df = pd.DataFrame({"x": xv.ravel(), "y": yv.ravel(), "z": z.ravel()})

    _chart = (
        alt.Chart(_df)
        .mark_rect()
        .encode(
            x=alt.X("x:O").axis(values=_df["x"][::5000]),
            y=alt.Y("y:O").axis(values=_df["y"][::5000]),
            color="z",
        )
        .properties(width=X * 2000, height=Y * 2000)
    )
    mo.ui.altair_chart(_chart)
    return


if __name__ == "__main__":
    app.run()
