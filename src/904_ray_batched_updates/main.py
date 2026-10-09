import os
import queue
import threading
import time

import gymnasium as gym
import numpy as np
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.metrics import WEIGHTS_SEQ_NO

HZ = 50


def make_module(config):
    return config.get_rl_module_spec(env=gym.make("CartPole-v1")).build()


def episode_to_message(ep, dist_inputs, logp):
    return {
        "type": "EPISODES_AND_GET_STATE",
        "episodes": [{
            "obs": ep.get_observations().tolist(),
            "actions": ep.get_actions().tolist(),
            "rewards": ep.get_rewards().tolist(),
            "is_terminated": bool(ep.is_terminated),
            "is_truncated": bool(ep.is_truncated),
            "action_dist_inputs": dist_inputs.tolist(),
            "action_logp": logp.tolist(),
        }],
        "env_steps": len(ep.actions),
    }


def message_to_episode(msg, env):
    e = msg["episodes"][0]
    ep = SingleAgentEpisode(observation_space=env.observation_space, action_space=env.action_space)
    ep.add_env_reset(observation=e["obs"][0], infos={})
    for t in range(len(e["actions"])):
        ep.add_env_step(
            observation=e["obs"][t + 1],
            action=e["actions"][t],
            reward=e["rewards"][t],
            infos={},
            terminated=e["is_terminated"] and t == len(e["actions"]) - 1,
            truncated=e["is_truncated"] and t == len(e["actions"]) - 1,
            extra_model_outputs={
                Columns.ACTION_DIST_INPUTS: np.array(e["action_dist_inputs"][t]),
                Columns.ACTION_LOGP: e["action_logp"][t],
                WEIGHTS_SEQ_NO: 0,
            },
        )
    return ep.to_numpy()


def actor(out_queue, in_queue):
    env = gym.make("CartPole-v1")
    module = make_module(PPOConfig())
    obs, _ = env.reset()
    ep = SingleAgentEpisode(observation_space=env.observation_space, action_space=env.action_space)
    ep.add_env_reset(observation=obs, infos={})
    dist_inputs_all, logp_all = [], []

    fd = os.timerfd_create(time.CLOCK_MONOTONIC)
    os.timerfd_settime(fd, initial=1.0 / HZ, interval=1.0 / HZ)
    while True:
        os.read(fd, 8)
        if not in_queue.empty():
            msg = in_queue.get()
            if msg["type"] == "SET_STATE":
                module.set_state(msg["rl_module"])
        dist_inputs = module.forward_inference(
            {Columns.OBS: torch.tensor(np.array([obs], dtype=np.float32))}
        )[Columns.ACTION_DIST_INPUTS].detach().numpy()[0]
        probs = np.exp(dist_inputs - dist_inputs.max())
        probs /= probs.sum()
        action = int(np.random.choice(len(probs), p=probs))
        logp = float(np.log(probs[action]))
        next_obs, reward, term, trunc, _ = env.step(action)
        ep.add_env_step(
            observation=next_obs, action=action, reward=reward, infos={}, terminated=term, truncated=trunc,
            extra_model_outputs={Columns.ACTION_DIST_INPUTS: dist_inputs, Columns.ACTION_LOGP: logp, WEIGHTS_SEQ_NO: 0},
        )
        dist_inputs_all.append(dist_inputs)
        logp_all.append(logp)
        if term or trunc:
            out_queue.put(episode_to_message(ep.to_numpy(), np.array(dist_inputs_all), np.array(logp_all)))
            print(f"actor: episode len {len(ep.actions)}", flush=True)
            obs, _ = env.reset()
            ep = SingleAgentEpisode(observation_space=env.observation_space, action_space=env.action_space)
            ep.add_env_reset(observation=obs, infos={})
            dist_inputs_all, logp_all = [], []
        else:
            obs = next_obs


def learner(config, in_queue, out_queue):
    lg = config.build_learner_group(env=gym.make("CartPole-v1"))
    env = gym.make("CartPole-v1")
    ts, seq = 0, 0
    while True:
        time.sleep(1.0)
        msgs = []
        while True:
            try:
                msgs.append(in_queue.get_nowait())
            except queue.Empty:
                break
        if not msgs:
            continue
        episodes = [message_to_episode(m, env) for m in msgs]
        ts += sum(m["env_steps"] for m in msgs)
        lg.update(episodes=episodes, timesteps={"num_env_steps_sampled_lifetime": ts})
        seq += 1
        state = lg.get_state(components=["learner"], inference_only=True)["learner"]["rl_module"]["default_policy"]
        out_queue.put({"type": "SET_STATE", "rl_module": state, WEIGHTS_SEQ_NO: seq})
        print(f"learner: updated on {ts} total steps, weights seq {seq}")


def main():
    config = PPOConfig().environment("CartPole-v1").api_stack(enable_rl_module_and_learner=True)
    to_learner, to_actor = queue.Queue(), queue.Queue()
    threading.Thread(target=actor, args=(to_learner, to_actor), daemon=True).start()
    learner(config, to_learner, to_actor)


if __name__ == "__main__":
    main()
