# Practice

## Getting started

```bash
git clone https://github.com/noahjutz-2026-wise/practice.git
git lfs pull
```

## Sim-to-real Quickstart

For details, see each README.

|     | README                                                 | Task                                                  |
| --- | ------------------------------------------------------ | ----------------------------------------------------- |
| 1   | [706_maze_build123d](src/706_maze_build123d/README.md) | `uv run export`                                       |
| 2   | [704_backend_mujoco](src/704_backend_mujoco/README.md) | `uv run build-xml`                                    |
| 3   | [705_backend_real](src/705_backend_real/README.md)     | `podman-compose up -d --build`                        |
| 4   | [701_pipeline_rllib](src/701_pipeline_rllib/README.md) | `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run train-mujoco` |

## Structure

Almost every directory under `src` is self-contained.

| Directories | Description                               |
| ----------- | ----------------------------------------- |
| 0xx         | Quick tool evaluations, early experiments |
| 7xx         | Prototypes                                |
| 9xx         | Entirely AI-Generated single-use code     |
