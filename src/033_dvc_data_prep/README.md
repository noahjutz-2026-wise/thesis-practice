# Metrics management with DVC + Parquet + Zenodo

## Quickstart

Fetching raw data: (not yet implemented)

```bash
dvc update -R . # fetch from zenodo via HTTP
```

Processing data:

```bash
uv run python -m dvc_data_prep.cam_latency.generate # out.pq
uv run python -m dvc_data_prep.cam_latency.process # out.csv
```
