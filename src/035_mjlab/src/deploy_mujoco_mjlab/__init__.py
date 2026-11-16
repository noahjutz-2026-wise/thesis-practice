from mjlab.tasks.registry import register_mjlab_task

from deploy_mujoco_mjlab.main import cartpole_ppo_runner_cfg, cartpole_swingup_env_cfg

register_mjlab_task(
    task_id="My-Mjlab-Cartpole-Swingup",
    env_cfg=cartpole_swingup_env_cfg(),
    play_env_cfg=cartpole_swingup_env_cfg(play=True),
    rl_cfg=cartpole_ppo_runner_cfg(),
)
