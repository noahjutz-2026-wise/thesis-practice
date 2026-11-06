import gymnasium as gym


class LabyrinthRealEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
