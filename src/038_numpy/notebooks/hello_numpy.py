import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell
def _(np):
    w = np.hstack((np.arange(10).reshape(-1,2) + 10, np.arange(5).reshape(-1, 1)))
    w
    return (w,)


@app.cell
def _(mo, w):
    v = w[:, [[0,2],[1,2]]]
    h = w[:, [[2,0],[2,1]]]

    mo.hstack([mo.plain_text(str(v)), mo.plain_text(str(h))])
    return (v,)


@app.cell
def _(v):
    v
    return


if __name__ == "__main__":
    app.run()
