from ray.rllib.algorithms.ppo import PPOConfig


def main():
    algo = (
        PPOConfig()
        .environment("CartPole-v1")
        .env_runners(num_env_runners=1, num_envs_per_env_runner=1)
        .learners(num_learners=1, num_gpus_per_learner=0)
        .build_algo()
    )

    while True:
        results = algo.train()
        print(results["env_runners"]["episode_return_mean"])


if __name__ == "__main__":
    main()
