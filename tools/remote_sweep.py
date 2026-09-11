#!/usr/bin/env python3
"""Launch up to four independent one-GPU training runs from a JSON plan."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("plan", type=Path)
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()
  plan = json.loads(args.plan.read_text())

  group = plan["group"]
  task = plan["task"]
  common_args = plan.get("common_args", [])
  runs = plan["runs"]
  if not SAFE_NAME.fullmatch(group):
    raise ValueError(f"Unsafe group name: {group}")
  if not 1 <= len(runs) <= 4:
    raise ValueError("A sweep must contain between one and four runs.")
  if any(arg.startswith("--gpu-ids") for arg in common_args):
    raise ValueError("Set GPU IDs per run, not in common_args.")
  if any(arg.startswith("--agent.seed") for arg in common_args):
    raise ValueError("Set the explicit seed in each run entry.")

  gpu_ids = [int(run["gpu"]) for run in runs]
  if len(gpu_ids) != len(set(gpu_ids)) or any(gpu not in range(4) for gpu in gpu_ids):
    raise ValueError("Each run must use a distinct GPU ID from 0 through 3.")

  project_root = Path(__file__).resolve().parents[1]
  sessions = []
  commands = []
  for run in runs:
    name = run["name"]
    if not SAFE_NAME.fullmatch(name):
      raise ValueError(f"Unsafe run name: {name}")
    session = f"{group}_{name}"
    run_args = run.get("args", [])
    if any(
      arg.startswith(("--gpu-ids", "--agent.seed")) for arg in run_args
    ):
      raise ValueError(f"Run {name} must use its gpu and seed fields.")
    sessions.append(session)
    command = [
      str(project_root / "tools" / "remote_train.sh"),
      session,
      task,
      "--gpu-ids",
      f"[{int(run['gpu'])}]",
      f"--agent.seed={int(run['seed'])}",
      *common_args,
      *run_args,
    ]
    commands.append(command)

  if args.dry_run:
    for command in commands:
      print(" ".join(command))
    return

  env = os.environ.copy()
  env["AUTOTUNE_NO_REGISTER"] = "1"
  launched = []
  try:
    for session, command in zip(sessions, commands, strict=True):
      subprocess.run(command, cwd=project_root, env=env, check=True)
      launched.append(session)
  finally:
    if launched:
      group_dir = project_root / ".autotune" / "groups"
      group_dir.mkdir(parents=True, exist_ok=True)
      record = {
        "group": group,
        "mode": (
          plan.get("mode", "screening")
          if len(launched) == len(sessions)
          else "partial-launch"
        ),
        "plan_file": str(args.plan.resolve()),
        "sessions": launched,
        "expected_sessions": sessions,
      }
      (group_dir / f"{group}.json").write_text(json.dumps(record, indent=2) + "\n")
  if len(launched) != len(sessions):
    raise RuntimeError(f"Only launched {len(launched)} of {len(sessions)} runs.")
  print(f"Registered watcher group: {group} ({len(sessions)} runs)")


if __name__ == "__main__":
  main()
