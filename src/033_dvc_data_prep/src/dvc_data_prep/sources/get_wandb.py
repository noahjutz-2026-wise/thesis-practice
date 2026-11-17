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
            if seed in seen_seeds:
                continue
            seen_seeds.add(seed)
            picked.append(run)
            if len(picked) == n:
                break
        out[tag] = picked
    return out


METRIC = "env_runners/episode_return_max"


def episode_returns():
    """tag -> list of the aggregate episode_return_max scalar per run."""
    out = {}
    for tag, runs in last_runs_by_tag().items():
        values = []
        for run in runs:
            value = run.summary.get(METRIC)
            if value is None:
                raise ValueError(
                    f"run {run.name} (tag {tag!r}, seed {run.config.get('seed')}) "
                    f"has no {METRIC} in its summary"
                )
            values.append(float(value))
        out[tag] = values
    return out
