# Metrics management with DVC + Parquet + Zenodo

## Quickstart

```bash
dvc update -R . # to get out.pq from zenodo
```

## Supporting scripts

```bash
uv run record-pq # to generate out.pq from camera
uv run read-pq # read to pandas, pyarrow or numpy
```
