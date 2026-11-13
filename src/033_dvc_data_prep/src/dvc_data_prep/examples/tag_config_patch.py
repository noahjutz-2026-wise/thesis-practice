import re
import warnings

import wandb

from dvc_data_prep.sources.get_wandb import tags

ENTITY_PROJECT = "tjno/agent-experiments"


def experiment_family(tag: str) -> str:
    return re.sub(r"-[^-]+$", "", tag)


def main() -> None:
    api = wandb.Api()

    for tag in tags:
        runs = list(
            api.runs(
                ENTITY_PROJECT,
                filters={
                    "tags": {"$in": [tag]},
                    "state": {"$ne": "crashed"},
                },
            )
        )

        if len(runs) != 5:
            warnings.warn(
                f"[{tag}] expected 5 runs, got {len(runs)} — skipping.",
                stacklevel=2,
            )
            continue

        family = experiment_family(tag)

        for run in runs:
            run.config["experiment_name"] = tag
            run.config["experiment_family"] = family
            run.update()
            print(f"  patched {run.id} ({run.name})")

        print(f"[{tag}] → family='{family}'  ({len(runs)} runs patched)")


if __name__ == "__main__":
    main()
