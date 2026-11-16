# MuJoCo Playground

## CartPole Example

```bash
uv run python -m mujoco_playground.rl.rsl_rl.train \
    --env=CartpoleBalance \
    --logdir=/tmp/rslrl_logs/cartpole_balance

uv run python -m mujoco_playground.rl.rsl_rl.play \
    --env=CartpoleBalance \
    --resume_path=/tmp/rslrl_logs/cartpole_balance
```
