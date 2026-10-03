#!/usr/bin/env python3
"""Validate EgoDex+PICO synthesized kneeling command/style pairs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


COMMAND_REQUIRED = (
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

STYLE_NAMES = (
  "pelvis_height_neutral_norm",
  "effective_leg_length_left",
  "effective_leg_length_right",
  "knee_ground_distance_left",
  "knee_ground_distance_right",
  "torso_pitch",
  "shoulder_mid_rel_pelvis_forward",
  "shoulder_mid_rel_pelvis_left",
  "shoulder_pelvis_yaw_difference",
  "left_foot_rel_pelvis_forward",
  "left_foot_rel_pelvis_left",
  "right_foot_rel_pelvis_forward",
  "right_foot_rel_pelvis_left",
  "foot_pseudo_contact_left",
  "foot_pseudo_contact_right",
  "knee_pseudo_contact_left",
  "knee_pseudo_contact_right",
)


def _metadata(data) -> dict:
  if "metadata_json" not in data.files:
    return {}
  raw = data["metadata_json"]
  if raw.ndim == 0:
    raw = raw.item()
  try:
    return json.loads(str(raw))
  except Exception:
    return {}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("root", type=Path)
  ap.add_argument("--show-files", action="store_true")
  args = ap.parse_args()

  root = args.root.expanduser().resolve()
  files = sorted(
    p for p in root.rglob("*.npz")
    if not p.name.endswith(".style.npz")
    and not p.name.endswith(".style_candidates.npz")
    and p.name != "kneel_template_library.npz"
  )

  sides = Counter()
  phase_sets = Counter()
  valid = 0
  invalid = 0
  candidate_counts: list[int] = []
  task_weight_ranges: list[tuple[float, float]] = []

  for path in files:
    try:
      with np.load(path, allow_pickle=False) as data:
        missing = [k for k in COMMAND_REQUIRED if k not in data.files]
        if missing:
          raise RuntimeError(f"missing command keys {missing}")
        ts = np.asarray(data["timestamp"], dtype=np.float64)
        if ts.ndim != 1 or len(ts) < 2 or np.any(np.diff(ts) < 0):
          raise RuntimeError("invalid timestamp")
        n = len(ts)
        md = _metadata(data)
        side = str(md.get("selected_pico_side", "unknown"))
        sides[side] += 1

        if "phase" in data.files:
          phase = np.asarray(data["phase"])
          if phase.shape != (n,):
            raise RuntimeError(f"phase shape {phase.shape} != {(n,)}")
          pset = tuple(int(x) for x in np.unique(phase))
        else:
          pset = ()
        phase_sets[pset] += 1

        if "wrist_tracking_weight_scale" in data.files:
          w = np.asarray(data["wrist_tracking_weight_scale"], dtype=float)
          if w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w < 0):
            raise RuntimeError("invalid wrist_tracking_weight_scale")
          task_weight_ranges.append((float(w.min()), float(w.max())))

      style_path = path.with_name(f"{path.stem}.style_candidates.npz")
      legacy_path = path.with_name(f"{path.stem}.style.npz")
      if style_path.is_file():
        with np.load(style_path, allow_pickle=False) as s:
          required = (
            "timestamp",
            "style_candidates",
            "style_candidate_valid",
            "style_descriptor_names",
          )
          missing = [k for k in required if k not in s.files]
          if missing:
            raise RuntimeError(f"{style_path.name}: missing {missing}")
          st = np.asarray(s["timestamp"], dtype=float)
          c = np.asarray(s["style_candidates"], dtype=float)
          v = np.asarray(s["style_candidate_valid"], dtype=bool)
          names = tuple(str(x) for x in s["style_descriptor_names"])
          if st.shape != (n,) or not np.allclose(
            st - st[0], ts - ts[0], atol=1e-5, rtol=0
          ):
            raise RuntimeError("candidate style timestamps are not synchronized")
          if c.ndim != 3 or c.shape[1:] != (n, len(STYLE_NAMES)):
            raise RuntimeError(f"style_candidates shape={c.shape}")
          if v.shape != c.shape[:2]:
            raise RuntimeError(f"style_candidate_valid shape={v.shape}")
          if names != STYLE_NAMES:
            raise RuntimeError("descriptor schema mismatch")
          candidate_counts.append(c.shape[0])
          idx = int(md.get("selected_pico_template_index", -1))
          if not 0 <= idx < c.shape[0]:
            raise RuntimeError(
              f"selected_pico_template_index={idx} outside [0,{c.shape[0]-1}]"
            )
      elif legacy_path.is_file():
        candidate_counts.append(1)
      else:
        raise RuntimeError("missing .style_candidates.npz/.style.npz")

      valid += 1
      if args.show_files:
        rel = path.relative_to(root)
        print(
          f"VALID {rel} T={n} side={side} "
          f"phase={pset} K={candidate_counts[-1]}"
        )
    except Exception as exc:
      invalid += 1
      print(f"INVALID {path}: {exc}")

  print()
  print(f"commands={len(files)} valid={valid} invalid={invalid}")
  print(f"selected_pico_side={dict(sides)}")
  print(f"phase_sets={dict(phase_sets)}")
  if candidate_counts:
    print(
      "style_candidates K: "
      f"min={min(candidate_counts)} median={np.median(candidate_counts):.1f} "
      f"max={max(candidate_counts)}"
    )
  if task_weight_ranges:
    print(
      "wrist_tracking_weight_scale: "
      f"global_min={min(x[0] for x in task_weight_ranges):.3f} "
      f"global_max={max(x[1] for x in task_weight_ranges):.3f}"
    )
  raise SystemExit(1 if invalid else 0)


if __name__ == "__main__":
  main()
