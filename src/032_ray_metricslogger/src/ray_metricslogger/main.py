from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType


class MyCallback(RLlibCallback):
    def on_episode_end(
        self,
        *,
        episode: Union[EpisodeType, EpisodeV2],
        prev_episode_chunks: Optional[List[EpisodeType]] = None,
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ) -> None:
        assert type(metrics_logger) == MetricsLogger
        assert type(episode) == SingleAgentEpisode
        metrics_logger.log_value(("termination", "success"), 1, reduce="sum")


def main():
    algo = (
        PPOConfig()
        .environment("CartPole-v1")
        .env_runners(num_env_runners=1, num_envs_per_env_runner=1)
        .learners(num_learners=1, num_gpus_per_learner=0)
        .callbacks(MyCallback)
        .build_algo()
    )

    while True:
        results = algo.train()
        print(results["env_runners"]["termination"]["success"])


if __name__ == "__main__":
    main()
