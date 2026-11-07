import argparse
import json

import wandb

CHECKPOINT_EVAL_WANDB_STEP_METRIC = "checkpoint_eval/checkpoint_step"


def _define_metrics_from_payload(metrics: dict[str, object]) -> None:
    step_metrics = [
        key for key in metrics if str(key).endswith("/checkpoint_step") or key == CHECKPOINT_EVAL_WANDB_STEP_METRIC
    ]
    for step_metric in step_metrics:
        wandb.define_metric(step_metric)
    for metric_name in metrics:
        if metric_name in step_metrics:
            continue
        prefix, _, leaf = str(metric_name).rpartition("/")
        if not prefix:
            continue
        step_metric = f"{prefix}/checkpoint_step"
        if step_metric not in metrics and step_metric != CHECKPOINT_EVAL_WANDB_STEP_METRIC:
            wandb.define_metric(step_metric)
        wandb.define_metric(metric_name, step_metric=step_metric)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--project", default="")
    parser.add_argument("--entity", default="")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--payload-json", default="")
    args = parser.parse_args()

    metrics = json.loads(args.metrics_json)
    if args.step is not None and CHECKPOINT_EVAL_WANDB_STEP_METRIC not in metrics:
        metrics[CHECKPOINT_EVAL_WANDB_STEP_METRIC] = int(args.step)
    payload = json.loads(args.payload_json) if args.payload_json else None

    run = wandb.init(
        project=args.project or None,
        entity=args.entity or None,
        id=args.run_id,
        resume="allow",
    )
    _define_metrics_from_payload(metrics)
    # Keep checkpoint_step in the payload as the custom x-axis metric instead of
    # forcing W&B's global step backward when checkpoint eval finishes after the
    # trainer has already advanced.
    wandb.log(metrics)
    run.finish()
    if payload is not None:
        print(json.dumps(payload))


if __name__ == "__main__":
    main()
