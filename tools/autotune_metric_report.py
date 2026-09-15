#!/usr/bin/env python3
"""Create a deterministic, no-LLM email report from remote TensorBoard values."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")
KEEP_RAW = (
  "Train/mean_reward", "Train/mean_episode_length", "Episode_Metrics/mean_action_acc",
  "Episode_Reward/action_rate", "Episode_Reward/leg_joint_acc",
  "Episode_Reward/feet_slide", "Episode_Reward/self_collisions",
  "Metrics/wrists/leg_joint_acc_rms_snapshot", "Metrics/wrists/torso_backward_lean",
  "Metrics/wrists/persistent_fraction", "Metrics/wrists/persistent_steady_fraction",
  "Metrics/wrists/continuous_fraction", "Metrics/wrists/continuous_moving_fraction",
  "Metrics/wrists/diag_contact_fraction", "Metrics/wrists/diag_release_events_masked",
  "Metrics/wrists/diag_recoveries_masked",
)


def decode_many(text: str) -> list[dict]:
  decoder, pos, values = json.JSONDecoder(), 0, []
  while pos < len(text):
    while pos < len(text) and text[pos].isspace(): pos += 1
    if pos >= len(text): break
    value, pos = decoder.raw_decode(text, pos)
    values.append(value)
  return values


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument("--group-file", type=Path)
  source.add_argument("--session")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--window", type=int, default=500)
  args = parser.parse_args()
  if args.window < 1: parser.error("--window must be positive")
  if args.group_file:
    group = json.loads(args.group_file.read_text())
    sessions = group["sessions"]
    title = group["group"]
  else:
    sessions, title = [args.session], args.session
  if not sessions or any(not SAFE.fullmatch(name or "") for name in sessions):
    raise ValueError("Unsafe or empty session list")
  command = [
    "ssh", "unitree-trainer",
    "/home/dev/miniconda3/envs/unitree_rl_mjlab/bin/python",
    "/home/dev/unitree_rl_mjlab/tools/summarize_wrist_training.py",
    *sessions, "--window", str(args.window),
  ]
  proc = subprocess.run(command, check=True, text=True, capture_output=True)
  values = decode_many(proc.stdout)
  if len(values) != len(sessions):
    raise RuntimeError(f"Expected {len(sessions)} summaries, got {len(values)}")
  report = {
    "title": title,
    "kind": "deterministic TensorBoard training diagnostics; no LLM analysis and no held-out rollout",
    "window": args.window,
    "sessions": {},
  }
  for value in values:
    session = value["session"]
    exit_code = subprocess.run(
      ["ssh", "unitree-trainer", "cat", f"/home/dev/unitree_rl_mjlab/.autotune/{session}/exit_code"],
      check=True, text=True, capture_output=True,
    ).stdout.strip()
    raw = value["raw_means"]
    report["sessions"][session] = {
      "exit_code": exit_code,
      "log_dir": value.get("log_dir"),
      "latest_checkpoint": value.get("latest_checkpoint"),
      "conditional": value["conditional"],
      "selected_raw": {key: raw[key] for key in KEEP_RAW if key in raw},
      "termination": {key: val for key,val in raw.items() if key.startswith("Episode_Termination/")},
      "nonfinite_tags": value["nonfinite_tags"],
    }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
  print(args.output)


if __name__ == "__main__": main()
