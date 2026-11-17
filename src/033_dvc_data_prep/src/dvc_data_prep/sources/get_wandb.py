import wandb

tags = [
    "baseline",
    "obs-holes-1",
    "obs-holes-2",
    "obs-holes-3",
    "obs-holes-10",
    "obs-walls-ray5",
    "obs-walls-ray15",
    "obs-walls-pts20",
    "obs-tvec-t0-False",
    "obs-tvec-t0-True",
    "obs-tvec-t002-False",
    "obs-tvec-t002-True",
    "obs-tvec-tdefault-False",
    "obs-tvec-tdefault-True",
    "obs-tvec-tfar-False",
    "obs-tvec-tfar-True",
    "obs-tvec-tmid-False",
    "obs-tvec-tmid-True",
    "obs-img-ball-rgb",
    "obs-img-ball-gray",
    "obs-img-top-rgb",
    "obs-img-top-gray",
    "obs-img-ball-gray-vec",
    "obs-img-top-gray-vec",
    "rew-const-0.0",
    "rew-const-0.0001",
    "rew-const-1e-05",
    "rew-holes-1e-05",
    "rew-holes-0.001",
    "rew-holes-0.1",
    "rew-negdelta-0.0",
    "rew-negdelta-0.1",
    "rew-clamp-0.01",
    "rew-clamp-0.1",
    "rew-clamp-1.0",
    "rew-clampdelta-0.01",
    "rew-clampdelta-0.1",
    "rew-clampdelta-1.0",
    "rl-hpo-lr-1e-3",
    "rl-hpo-lr-1e-5",
    "rl-hpo-ent-0.001",
    "rl-hpo-ent-0.1",
    "rl-hpo-clip-0.05",
    "rl-hpo-clip-0.4",
    "combo-tvec002-holes10",
    "combo-tvec002-wallsray5",
    "combo-tvecdef-negdelta01",
    "mujoco-imu",
    "curric-nhw-nh-normal",
    "curric-nhw-nw-normal",
]

ENTITY_PROJECT = "tjno/agent-experiments"


def agent_experiments(tag):
    api = wandb.Api()
    runs = api.runs(ENTITY_PROJECT, filters={"tags": tag})
    return runs


def last_runs_by_tag(n=5):
    """For each tag, the n most recent runs with distinct seeds."""
    api = wandb.Api()
    out = {}
    for tag in tags:
        runs = list(api.runs(ENTITY_PROJECT, filters={"tags": tag}))
        if len(runs) < n:
            raise ValueError(f"tag {tag!r}: only {len(runs)} runs, expected >= {n}")
        runs.sort(key=lambda r: r.created_at, reverse=True)
        picked, seen_seeds = [], set()
        for run in runs:
            seed = run.config.get("seed")
            if seed is None:
                # curric-* runs don't log a config seed; it's in the run name
                name_seed = run.name.rsplit("_seed", 1)
                if len(name_seed) == 2 and name_seed[1].isdigit():
                    seed = int(name_seed[1])
            if seed in seen_seeds:
                continue
            seen_seeds.add(seed)
            picked.append(run)
            if len(picked) == n:
                break
        out[tag] = picked
    return out


METRIC = "env_runners/episode_return_max"
FALLBACK_METRIC = "train/return_mean"


def episode_returns():
    """tag -> list of the aggregate episode_return_max scalar per run."""
    out = {}
    for tag, runs in last_runs_by_tag().items():
        values = []
        for run in runs:
            value = run.summary.get(METRIC)
            if value is None:
                # curric-* runs log train/return_mean instead of env_runners/*
                value = run.summary.get(FALLBACK_METRIC) if "curric-" in tag else None
            if value is None:
                raise ValueError(
                    f"run {run.name} (tag {tag!r}, seed {run.config.get('seed')}) "
                    f"has no {METRIC} in its summary"
                )
            values.append(float(value))
        out[tag] = values
    return out
