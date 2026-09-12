#!/usr/bin/env python3
"""Read TensorBoard and correctly normalize masked wrist command metrics."""

import argparse
import json
import math
import statistics
from pathlib import Path


def conditional_metrics(means: dict[str, float]) -> dict[str, float | None]:
  prefix = "Metrics/wrists/"
  fraction = means[prefix + "height_command_fraction"]
  if not 0.0 <= fraction <= 1.0:
    raise ValueError(f"Invalid height command fraction: {fraction}")
  result = {}
  for name, denominator in (
    ("height_wrist_pos_error", fraction),
    ("height_shoulder_error", fraction),
    ("height_target_vertical_gap", fraction),
    ("height_final_shoulder_target", fraction),
    ("height_final_wrist_target", fraction),
    ("nonheight_wrist_pos_error", 1.0 - fraction),
  ):
    key = prefix + name + "_masked"
    if key not in means:
      key = prefix + name  # Historical v2 names were also masked numerators.
    if key in means:
      result[name] = means[key] / denominator if denominator > 0.0 else None
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("sessions", nargs="+")
  parser.add_argument("--window", type=int, default=100)
  args = parser.parse_args()
  if args.window < 1:
    parser.error("--window must be positive")
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  root = Path(__file__).resolve().parents[1]
  for session in args.sessions:
    if Path(session).name != session or session in (".", ".."):
      parser.error(f"Unsafe session name: {session}")
    metadata = json.loads((root / ".autotune" / session / "run_metadata.json").read_text())
    events = EventAccumulator(metadata["log_dir"], size_guidance={"scalars": 0}).Reload()
    means = {}
    counts = {}
    nonfinite = []
    for tag in events.Tags()["scalars"]:
      if tag.startswith(("Metrics/wrists/", "Metrics/twist/", "Episode_Termination/")) or tag in (
        "Train/mean_episode_length", "Train/mean_reward", "Episode_Metrics/mean_action_acc"
      ):
        values = events.Scalars(tag)
        counts[tag] = len(values)
        if any(not math.isfinite(value.value) for value in values):
          nonfinite.append(tag)
        if values:
          means[tag] = statistics.fmean(value.value for value in values[-args.window:])
    print(json.dumps({
      "session": session,
      "window": args.window,
      "metric_scope": "command reset snapshots; training diagnostics, not held-out evaluation",
      "conditional": conditional_metrics(means),
      "raw_means": means,
      "sample_counts": counts,
      "nonfinite_tags": nonfinite,
    }, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
  main()
