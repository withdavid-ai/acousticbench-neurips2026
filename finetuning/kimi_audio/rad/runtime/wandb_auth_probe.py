import json
import os

import wandb


def main() -> None:
    api = wandb.Api(api_key=os.environ["WANDB_API_KEY"])
    viewer = api.viewer
    print(
        json.dumps(
            {
                "viewer": getattr(viewer, "username", None)
                or getattr(viewer, "entity", None),
                "project": os.environ.get("WANDB_PROJECT"),
                "entity": os.environ.get("WANDB_ENTITY"),
                "base_url": os.environ.get("WANDB_BASE_URL") or "https://wandb.ai",
                "mode": os.environ.get("WANDB_MODE"),
            }
        )
    )


if __name__ == "__main__":
    main()
