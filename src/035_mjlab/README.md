# mjlab

## Quickstart

```bash
uv run play Labyrinth-Game-v0 --agent random # View the environment
uv run train Labyrinth-Game-v0 # not yet implemented
uv run play Labyrinth-Game-v0 --wandb-run-path entity/project/run # not yet implemented
```

## Debugging

Documentation: [Viewers](https://mujocolab.github.io/mjlab/v1.4.0/source/viewers.html)

## Misc

```bash
uv run python -m deploy_mujoco_mjlab.labyrinth # Manual invocation of NativeMujocoViewer
```
