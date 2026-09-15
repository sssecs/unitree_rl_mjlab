#!/usr/bin/env python3
"""Render compact TensorBoard training curves for completion email."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/g1-autotune-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PANELS = (
  ("Train/mean_reward", "Mean reward"),
  ("Train/mean_episode_length", "Episode length"),
  ("Metrics/wrists/wrist_pos_error_mean", "Wrist position error (m)"),
  ("Metrics/wrists/wrist_rot_error_mean", "Wrist orientation error (rad)"),
  ("Episode_Metrics/mean_action_acc", "Mean action acceleration"),
  ("Episode_Termination/bad_orientation", "Bad-orientation termination"),
)


def smooth(values: np.ndarray, width: int = 51) -> np.ndarray:
  if len(values) < 3:
    return values
  width = min(width, len(values) if len(values) % 2 else len(values) - 1)
  if width < 3:
    return values
  kernel = np.ones(width) / width
  return np.convolve(values, kernel, mode="same")


def load(session: str) -> tuple[str, EventAccumulator]:
  root = Path(__file__).resolve().parents[1]
  if Path(session).name != session or session in (".", ".."):
    raise ValueError(f"Unsafe session: {session}")
  metadata = json.loads((root / ".autotune" / session / "run_metadata.json").read_text())
  return metadata["log_dir"], EventAccumulator(metadata["log_dir"], size_guidance={"scalars": 0}).Reload()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("sessions", nargs="+")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  runs = [(session, *load(session)) for session in args.sessions]
  figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
  for axis, (tag, label) in zip(axes.flat, PANELS):
    any_data = False
    for session, _, events in runs:
      if tag not in events.Tags()["scalars"]:
        continue
      values = events.Scalars(tag)
      x = np.asarray([value.step for value in values])
      y = np.asarray([value.value for value in values])
      axis.plot(x, smooth(y), linewidth=1.4, label=session)
      any_data = True
    axis.set(title=label, xlabel="PPO iteration")
    axis.grid(alpha=.25)
    if not any_data:
      axis.text(.5, .5, "not logged", ha="center", va="center", transform=axis.transAxes)
  if len(runs) > 1:
    axes.flat[0].legend(fontsize=7)
  figure.suptitle("G1 training diagnostics — TensorBoard curves (smoothed)", fontsize=13)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(args.output, dpi=170)
  print(args.output)


if __name__ == "__main__":
  main()
