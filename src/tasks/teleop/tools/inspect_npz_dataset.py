#!/usr/bin/env python3
"""Recursively validate sparse-command NPZs before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED = (
  "timestamp",
  "left_wrist_pos_world",
  "left_wrist_quat_world_wxyz",
  "right_wrist_pos_world",
  "right_wrist_quat_world_wxyz",
  "torso_center_xy_world",
  "left_shoulder_height",
  "right_shoulder_height",
  "torso_heading_world",
)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("root", type=Path)
  parser.add_argument(
    "--show-files",
    action="store_true",
    help="Print every valid file and its duration.",
  )
  args = parser.parse_args()

  root = args.root.expanduser().resolve()
  files = sorted(root.rglob("*.npz"))

  valid = 0
  invalid = 0
  total_frames = 0
  total_seconds = 0.0
  durations: list[float] = []
  versions: dict[str, int] = {}

  for path in files:
    try:
      with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED if key not in data.files]
        if missing:
          raise RuntimeError(f"missing keys {missing}")

        ts = np.asarray(data["timestamp"], dtype=np.float64)
        if ts.ndim != 1 or len(ts) < 2:
          raise RuntimeError("timestamp needs >=2 samples")
        if not np.all(np.isfinite(ts)):
          raise RuntimeError("timestamp contains NaN/Inf")
        if np.any(np.diff(ts) < 0):
          raise RuntimeError("timestamp is not monotonic")

        duration = float(ts[-1] - ts[0])
        if duration <= 0:
          raise RuntimeError("non-positive duration")

        version = "unknown"
        if "metadata_json" in data.files:
          raw = data["metadata_json"]
          if raw.ndim == 0:
            raw = raw.item()
          try:
            version = str(
              json.loads(str(raw)).get("interface_version", "unknown")
            )
          except Exception:
            pass

      valid += 1
      total_frames += len(ts)
      total_seconds += duration
      durations.append(duration)
      versions[version] = versions.get(version, 0) + 1

      if args.show_files:
        rel = path.relative_to(root) if root.is_dir() else path.name
        print(f"VALID  {rel}  frames={len(ts)}  duration={duration:.3f}s")

    except Exception as exc:
      invalid += 1
      print(f"INVALID {path}: {exc}")

  print()
  print(f"found={len(files)} valid={valid} invalid={invalid}")
  print(f"frames={total_frames} duration={total_seconds:.1f}s")
  if durations:
    print(
      "clip_duration_s: "
      f"min={min(durations):.3f} "
      f"median={float(np.median(durations)):.3f} "
      f"max={max(durations):.3f}"
    )
  print(f"versions={versions}")


if __name__ == "__main__":
  main()
