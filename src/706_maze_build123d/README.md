# Maze CAD

## Quickstart

```bash
uv run export
```

## Development

```bash
uv run server
```

Visit http://localhost:32323

## More info

Manually decompose to convex parts:

```bash
uv run coacd -i part0.stl -o floor.stl -t 0.005 --split
```
