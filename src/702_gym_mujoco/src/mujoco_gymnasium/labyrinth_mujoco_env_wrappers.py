from gymnasium.wrappers import TransformObservation

from mujoco_gymnasium.labyrinth_mujoco_env import LabyrinthMujocoEnv


def vec_obs_env(env: LabyrinthMujocoEnv):
    """
    Filter only vector part of obs
    """
    return TransformObservation(
        env, lambda obs: obs["vec"], env.observation_space["vec"]
    )


def img_obs_env(env: LabyrinthMujocoEnv):
    """
    Filter only image part of obs
    """
    return TransformObservation(
        env, lambda obs: obs["subimg"], env.observation_space["subimg"]
    )
