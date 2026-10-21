# Maze CAD

## Quickstart

```bash
uv run --with watchfiles watchfiles "python src/labyrinth_game_cad/server.py"
```

Decompose to convex parts:

```bash
uv run coacd -i part0.stl -o test.stl -t 0.005
```
