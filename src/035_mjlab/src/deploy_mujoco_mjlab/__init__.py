from mjlab.tasks.registry import register_mjlab_task

from deploy_mujoco_mjlab.labyrinth import labyrinth_env_cfg, labyrinth_ppo_runner_cfg

# Baseline: untouched gym-env mirror (see report, run 1).
register_mjlab_task(
    task_id="My-Mjlab-Labyrinth",
    env_cfg=labyrinth_env_cfg(),
    play_env_cfg=labyrinth_env_cfg(play=True),
    rl_cfg=labyrinth_ppo_runner_cfg(experiment_name="labyrinth"),
)

# v2: explicit -1 fall penalty (baseline episodes end ~190 steps in, all falls).
register_mjlab_task(
    task_id="My-Mjlab-Labyrinth-v2-fallpen",
    env_cfg=labyrinth_env_cfg(fall_penalty_weight=-1.0),
    play_env_cfg=labyrinth_env_cfg(play=True, fall_penalty_weight=-1.0),
    rl_cfg=labyrinth_ppo_runner_cfg(experiment_name="labyrinth-v2-fallpen"),
)

# v3: v2 + half-strength board tilts (baseline policy std explodes to ~2,
# full-strength tilts hurl the ball into the first hole).
register_mjlab_task(
    task_id="My-Mjlab-Labyrinth-v3-gentle",
    env_cfg=labyrinth_env_cfg(fall_penalty_weight=-1.0, action_scale=0.5),
    play_env_cfg=labyrinth_env_cfg(play=True, fall_penalty_weight=-1.0, action_scale=0.5),
    rl_cfg=labyrinth_ppo_runner_cfg(experiment_name="labyrinth-v3-gentle"),
)

# v4: v3 + calm exploration (init_std 0.5, no entropy bonus).
register_mjlab_task(
    task_id="My-Mjlab-Labyrinth-v4-calm",
    env_cfg=labyrinth_env_cfg(fall_penalty_weight=-1.0, action_scale=0.5),
    play_env_cfg=labyrinth_env_cfg(play=True, fall_penalty_weight=-1.0, action_scale=0.5),
    rl_cfg=labyrinth_ppo_runner_cfg(
        experiment_name="labyrinth-v4-calm",
        entropy_coef=0.0,
        init_std=0.5,
    ),
)

# v5: OFAT-informed combo — v4 plus: no constant time penalty (the gym OFAT
# campaign found it measurably hurts: +61% return when removed) and normalized
# tvec lookahead (best single obs change there: +85% return).
register_mjlab_task(
    task_id="My-Mjlab-Labyrinth-v5-ofat",
    env_cfg=labyrinth_env_cfg(
        fall_penalty_weight=-1.0,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
    ),
    play_env_cfg=labyrinth_env_cfg(
        play=True,
        fall_penalty_weight=-1.0,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
    ),
    rl_cfg=labyrinth_ppo_runner_cfg(
        experiment_name="labyrinth-v5-ofat",
        entropy_coef=0.0,
        init_std=0.5,
    ),
)

# v6: v5 plus dense approach shaping (negative distance to the 0.02-lookahead
# target) — the OFAT report's "denser shaping along waypoints" recommendation.
register_mjlab_task(
    task_id="My-Mjlab-Labyrinth-v6-shaping",
    env_cfg=labyrinth_env_cfg(
        fall_penalty_weight=-1.0,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
    ),
    play_env_cfg=labyrinth_env_cfg(
        play=True,
        fall_penalty_weight=-1.0,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
    ),
    rl_cfg=labyrinth_ppo_runner_cfg(
        experiment_name="labyrinth-v6-shaping",
        entropy_coef=0.0,
        init_std=0.5,
    ),
)

# v7: FUNDAMENTAL rebuild of the incentive structure — start past the first
# hole (wp 11; wp 6 is 1 cm from a hole), hole-aware obs (3 nearest holes),
# goal bonus +50 vs fall penalty -5 (sitting still no longer dominates),
# no time penalty, normalized tvec, approach shaping. Proves the pipeline can
# produce goal terminations at all.
register_mjlab_task(
    task_id="My-Mjlab-Labyrinth-v7-fundamental",
    env_cfg=labyrinth_env_cfg(
        fall_penalty_weight=-5.0,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
        holes_obs_k=3,
        goal_bonus_weight=50.0,
        start_wp_idx=11,
    ),
    play_env_cfg=labyrinth_env_cfg(
        play=True,
        fall_penalty_weight=-5.0,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
        holes_obs_k=3,
        goal_bonus_weight=50.0,
        start_wp_idx=11,
    ),
    rl_cfg=labyrinth_ppo_runner_cfg(
        experiment_name="labyrinth-v7-fundamental",
        entropy_coef=0.0,
        init_std=0.5,
    ),
)

# v8: v7 but the full maze from the original start — the actual target.
register_mjlab_task(
    task_id="My-Mjlab-Labyrinth-v8-full",
    env_cfg=labyrinth_env_cfg(
        fall_penalty_weight=-5.0,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
        holes_obs_k=3,
        goal_bonus_weight=50.0,
    ),
    play_env_cfg=labyrinth_env_cfg(
        play=True,
        fall_penalty_weight=-5.0,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
        holes_obs_k=3,
        goal_bonus_weight=50.0,
    ),
    rl_cfg=labyrinth_ppo_runner_cfg(
        experiment_name="labyrinth-v8-full",
        entropy_coef=0.0,
        init_std=0.5,
    ),
)

# v9: deterministic start on the final straight segment (wp69), 3 waypoints
# from the goal, no holes within 24mm. Proves the reward/termination/goal
# wiring end-to-end; the start then walks backward across runs.
register_mjlab_task(
    task_id="My-Mjlab-Labyrinth-v9-endstretch",
    env_cfg=labyrinth_env_cfg(
        fall_penalty_weight=-5.0,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
        holes_obs_k=3,
        goal_bonus_weight=50.0,
        start_wp_idx=69,
    ),
    play_env_cfg=labyrinth_env_cfg(
        play=True,
        fall_penalty_weight=-5.0,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
        holes_obs_k=3,
        goal_bonus_weight=50.0,
        start_wp_idx=69,
    ),
    rl_cfg=labyrinth_ppo_runner_cfg(
        experiment_name="labyrinth-v9-endstretch",
        entropy_coef=0.0,
        init_std=0.5,
    ),
)

# v10: full maze from the original start, on the EXACT (non-COACD) mesh —
# the pocket/seam traps are gone; this is the real test.
register_mjlab_task(
    task_id="My-Mjlab-Labyrinth-v10-fixedmesh-full",
    env_cfg=labyrinth_env_cfg(
        fall_penalty_weight=-5.0,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
        holes_obs_k=3,
        goal_bonus_weight=50.0,
    ),
    play_env_cfg=labyrinth_env_cfg(
        play=True,
        fall_penalty_weight=-5.0,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
        holes_obs_k=3,
        goal_bonus_weight=50.0,
    ),
    rl_cfg=labyrinth_ppo_runner_cfg(
        experiment_name="labyrinth-v10-fixedmesh-full",
        entropy_coef=0.0,
        init_std=0.5,
    ),
)

# v11: v10 + observations expressed in the BOARD-TILT (action) frame.
# Empirically ctrl_x=+ rolls the ball toward -y and ctrl_y=+ toward +x, so a
# policy given world-frame vectors must learn a 90-degree rotated action map.
# Rotating ball velocity, tvec and hole vectors by 90 deg makes the optimal
# action map the identity.
register_mjlab_task(
    task_id="My-Mjlab-Labyrinth-v11-ctrlframe",
    env_cfg=labyrinth_env_cfg(
        fall_penalty_weight=-5.0,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
        holes_obs_k=3,
        goal_bonus_weight=50.0,
        ctrl_frame_obs=True,
    ),
    play_env_cfg=labyrinth_env_cfg(
        play=True,
        fall_penalty_weight=-5.0,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
        holes_obs_k=3,
        goal_bonus_weight=50.0,
        ctrl_frame_obs=True,
    ),
    rl_cfg=labyrinth_ppo_runner_cfg(
        experiment_name="labyrinth-v11-ctrlframe",
        entropy_coef=0.0,
        init_std=0.5,
    ),
)

# v12: v11 (ctrl-frame obs) on the CORRECTED exact mesh (first exact-mesh
# attempt was broken: earcut indices were used as coordinates, leaving no
# floor under the spawn -> free fall).
register_mjlab_task(
    task_id="My-Mjlab-Labyrinth-v12-exactmesh",
    env_cfg=labyrinth_env_cfg(
        fall_penalty_weight=-5.0,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
        holes_obs_k=3,
        goal_bonus_weight=50.0,
        ctrl_frame_obs=True,
    ),
    play_env_cfg=labyrinth_env_cfg(
        play=True,
        fall_penalty_weight=-5.0,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
        holes_obs_k=3,
        goal_bonus_weight=50.0,
        ctrl_frame_obs=True,
    ),
    rl_cfg=labyrinth_ppo_runner_cfg(
        experiment_name="labyrinth-v12-exactmesh",
        entropy_coef=0.0,
        init_std=0.5,
    ),
)

# v13: reward-rescale so progress dominates risk from the far start: at
# gamma=0.99 the goal bonus at 1000 steps is worth ~0.002 — nothing — while
# the fall penalty is immediate, so a rational policy never leaves the start.
# progress x20 (max +20/episode), fall -1, goal +50, gamma 0.995.
register_mjlab_task(
    task_id="My-Mjlab-Labyrinth-v13-rescaled",
    env_cfg=labyrinth_env_cfg(
        fall_penalty_weight=-1.0,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
        holes_obs_k=3,
        goal_bonus_weight=50.0,
        ctrl_frame_obs=True,
        progress_weight=20.0,
    ),
    play_env_cfg=labyrinth_env_cfg(
        play=True,
        fall_penalty_weight=-1.0,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
        holes_obs_k=3,
        goal_bonus_weight=50.0,
        ctrl_frame_obs=True,
        progress_weight=20.0,
    ),
    rl_cfg=labyrinth_ppo_runner_cfg(
        experiment_name="labyrinth-v13-rescaled",
        entropy_coef=0.0,
        init_std=0.5,
        gamma=0.995,
    ),
)

# v14: respawn-on-fall (checkpoint respawn instead of episode-ending fall).
# Quasi-static analysis says the maze has NO hole-safe route; holes are only
# crossable with momentum, and failed crossings must not erase the episode.
register_mjlab_task(
    task_id="My-Mjlab-Labyrinth-v14-respawn",
    env_cfg=labyrinth_env_cfg(
        fall_penalty_weight=-1.0,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
        holes_obs_k=3,
        goal_bonus_weight=50.0,
        ctrl_frame_obs=True,
        progress_weight=20.0,
        episode_steps=3000,
        respawn=True,
    ),
    play_env_cfg=labyrinth_env_cfg(
        play=True,
        fall_penalty_weight=-1.0,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
        holes_obs_k=3,
        goal_bonus_weight=50.0,
        ctrl_frame_obs=True,
        progress_weight=20.0,
        episode_steps=3000,
        respawn=True,
    ),
    rl_cfg=labyrinth_ppo_runner_cfg(
        experiment_name="labyrinth-v14-respawn",
        entropy_coef=0.0,
        init_std=0.5,
        gamma=0.998,
        max_iterations=500,
    ),
)

# v15: respawn + 60 s episodes + near-free falls. Careful traversal of the
# 1.95 m path with momentum hole-crossings needs ~40-100 s; the 18 s cap of
# v14 cannot contain it and gamma^3000 erases the goal bonus. Falls cost
# only time (respawn penalty -0.01 keeps the respawn term active).
register_mjlab_task(
    task_id="My-Mjlab-Labyrinth-v15-respawn-long",
    env_cfg=labyrinth_env_cfg(
        fall_penalty_weight=-0.01,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
        holes_obs_k=3,
        goal_bonus_weight=50.0,
        ctrl_frame_obs=True,
        progress_weight=20.0,
        episode_steps=10000,
        respawn=True,
    ),
    play_env_cfg=labyrinth_env_cfg(
        play=True,
        fall_penalty_weight=-0.01,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
        holes_obs_k=3,
        goal_bonus_weight=50.0,
        ctrl_frame_obs=True,
        progress_weight=20.0,
        episode_steps=10000,
        respawn=True,
    ),
    rl_cfg=labyrinth_ppo_runner_cfg(
        experiment_name="labyrinth-v15-respawn-long",
        entropy_coef=0.0,
        init_std=0.5,
        gamma=0.999,
        max_iterations=500,
    ),
)

# v16: v15 env + sustained exploration. v14/v15 both showed early progress
# decaying to zero as the policy std collapsed (entropy_coef 0 was tuned on
# short-horizon tasks; the full maze needs sustained exploration).
register_mjlab_task(
    task_id="My-Mjlab-Labyrinth-v16-respawn-explore",
    env_cfg=labyrinth_env_cfg(
        fall_penalty_weight=-0.01,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
        holes_obs_k=3,
        goal_bonus_weight=50.0,
        ctrl_frame_obs=True,
        progress_weight=20.0,
        episode_steps=10000,
        respawn=True,
    ),
    play_env_cfg=labyrinth_env_cfg(
        play=True,
        fall_penalty_weight=-0.01,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
        holes_obs_k=3,
        goal_bonus_weight=50.0,
        ctrl_frame_obs=True,
        progress_weight=20.0,
        episode_steps=10000,
        respawn=True,
    ),
    rl_cfg=labyrinth_ppo_runner_cfg(
        experiment_name="labyrinth-v16-respawn-explore",
        entropy_coef=0.01,
        init_std=1.0,
        gamma=0.999,
        max_iterations=500,
    ),
)

# v18: v15 env + tamer PPO. The ratchet runs show early progress followed by
# a cliff collapse with the adaptive LR pegged at 0.01 (10x base); halve the
# KL target and drop the base LR to keep late-training updates stable.
register_mjlab_task(
    task_id="My-Mjlab-Labyrinth-v18-respawn-stable",
    env_cfg=labyrinth_env_cfg(
        fall_penalty_weight=-0.01,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
        holes_obs_k=3,
        goal_bonus_weight=50.0,
        ctrl_frame_obs=True,
        progress_weight=20.0,
        episode_steps=10000,
        respawn=True,
    ),
    play_env_cfg=labyrinth_env_cfg(
        play=True,
        fall_penalty_weight=-0.01,
        action_scale=0.5,
        time_penalty_weight=0.0,
        normalize_tvec=True,
        approach_weight=0.5,
        holes_obs_k=3,
        goal_bonus_weight=50.0,
        ctrl_frame_obs=True,
        progress_weight=20.0,
        episode_steps=10000,
        respawn=True,
    ),
    rl_cfg=labyrinth_ppo_runner_cfg(
        experiment_name="labyrinth-v18-respawn-stable",
        entropy_coef=0.0,
        init_std=0.5,
        gamma=0.999,
        max_iterations=500,
        lr=3.0e-4,
        desired_kl=0.005,
    ),
)
# training with LONG episodes (3000 steps = 18 s, gamma 0.998) — the 6 s
# gym-episode cap made careful traversal unaffordable (long-66: 19.5
# goals/iter vs 0 with 6 s episodes). Final rung (0) is the deliverable.
for _wp in (60, 50, 40, 30, 20, 10, 0):
    register_mjlab_task(
        task_id=f"My-Mjlab-Labyrinth-ladder{_wp}",
        env_cfg=labyrinth_env_cfg(
            fall_penalty_weight=-1.0,
            action_scale=0.5,
            time_penalty_weight=0.0,
            normalize_tvec=True,
            approach_weight=0.5,
            holes_obs_k=3,
            goal_bonus_weight=50.0,
            ctrl_frame_obs=True,
            progress_weight=20.0,
            start_wp_idx=_wp,
            episode_steps=3000,
        ),
        play_env_cfg=labyrinth_env_cfg(
            play=True,
            fall_penalty_weight=-1.0,
            action_scale=0.5,
            time_penalty_weight=0.0,
            normalize_tvec=True,
            approach_weight=0.5,
            holes_obs_k=3,
            goal_bonus_weight=50.0,
            ctrl_frame_obs=True,
            progress_weight=20.0,
            start_wp_idx=_wp,
            episode_steps=3000,
        ),
        rl_cfg=labyrinth_ppo_runner_cfg(
            experiment_name=f"labyrinth-ladder{_wp}",
            entropy_coef=0.0,
            init_std=0.5,
            gamma=0.998,
            max_iterations=500,
        ),
    )

# Deep rungs: failing ladder starts retrained at 500 iterations; long
# episodes (3000 steps = 18 s) so careful traversal past holes is affordable.
for _wp in (66,):
    register_mjlab_task(
        task_id=f"My-Mjlab-Labyrinth-ladder{_wp}long",
        env_cfg=labyrinth_env_cfg(
            fall_penalty_weight=-1.0,
            action_scale=0.5,
            time_penalty_weight=0.0,
            normalize_tvec=True,
            approach_weight=0.5,
            holes_obs_k=3,
            goal_bonus_weight=50.0,
            ctrl_frame_obs=True,
            progress_weight=20.0,
            start_wp_idx=_wp,
            episode_steps=3000,
        ),
        play_env_cfg=labyrinth_env_cfg(
            play=True,
            fall_penalty_weight=-1.0,
            action_scale=0.5,
            time_penalty_weight=0.0,
            normalize_tvec=True,
            approach_weight=0.5,
            holes_obs_k=3,
            goal_bonus_weight=50.0,
            ctrl_frame_obs=True,
            progress_weight=20.0,
            start_wp_idx=_wp,
            episode_steps=3000,
        ),
        rl_cfg=labyrinth_ppo_runner_cfg(
            experiment_name=f"labyrinth-ladder{_wp}long",
            entropy_coef=0.0,
            init_std=0.5,
            gamma=0.998,
            max_iterations=500,
        ),
    )
