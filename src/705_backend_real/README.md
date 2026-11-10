# Backend Real

## Quickstart

```bash
# Start server
podman-compose up --build

# Interact
FILE_MARKERS_CSV=markers.csv \
    FILE_CAM_CONFIG_YAML=calibration.yaml \
    uv run python -m backend_real.client.debugger
uv run python -m backend_real.client.manual
```

# Development

```bash
uv sync --no-dev --group container # opencv-python
uv sync # opencv-python-headless
```
