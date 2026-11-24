import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import altair as alt
    import marimo as mo

    return alt, np, pd


@app.cell
def _(np):
    # Broadcasting
    _x = np.arange(100).reshape(-1, 2, 5, 2, 1)
    _y = np.arange(23)
    _shp = np.broadcast_shapes(_x.shape, _y.shape)
    _x.shape, _y.shape, _shp, (_x - _y).shape
    return


@app.cell
def _(np):
    # Vecdot
    _a = np.arange(1024).reshape(-1, 2, 2, 1)
    _b = np.arange(1024).reshape(-1, 8, 2, 1)

    np.vecdot(_a, _a)
    return


@app.cell
def _(alt, np, pd):
    # Plotting vectors
    a = np.random.random((8 * 8 * 2,))
    a = a * 2 - 1
    a = a.reshape(8, 8, -1, 2)
    H, W, N, _ = a.shape
    yy, xx, nn = np.meshgrid(np.arange(H), np.arange(W), np.arange(N), indexing="ij")
    df = pd.DataFrame({
        "x": xx.ravel(), "y": yy.ravel(), "n": nn.ravel(),
        "u": a[..., 0].ravel(), "v": a[..., 1].ravel(),
    })
    df["x2"] = df.x + df.u
    df["y2"] = df.y + df.v

    alt.Chart(df).mark_rule().encode(
        x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q",
        color="n:N",
    ).properties(
        width=500,
        height=500
    )
    return


if __name__ == "__main__":
    app.run()
