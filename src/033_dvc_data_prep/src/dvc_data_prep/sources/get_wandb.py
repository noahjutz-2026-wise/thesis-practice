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


def agent_experiments():
    api = wandb.Api()
    runs = api.runs("tjno/agent-experiments", filters={"tags": {"$in": tags}})
    return runs
