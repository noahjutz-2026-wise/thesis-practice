from mjlab.tasks.registry import register_mjlab_task

from deploy_mujoco_mjlab.labyrinth import labyrinth_env_cfg
from deploy_mujoco_mjlab.main import cartpole_ppo_runner_cfg

register_mjlab_task(
    task_id="My-Mjlab-Labyrinth",
    env_cfg=labyrinth_env_cfg(),
    play_env_cfg=labyrinth_env_cfg(play=True),
    rl_cfg=cartpole_ppo_runner_cfg(),
)
