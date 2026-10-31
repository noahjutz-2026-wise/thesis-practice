from ray.rllib.algorithms.ppo import PPOConfig


def main():
    base_config = PPOConfig()

    config = base_config.environment()
