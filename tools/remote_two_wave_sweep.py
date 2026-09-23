#!/usr/bin/env python3
"""Launch two bounded four-GPU sweep waves sequentially from the workstation."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


HOST = "unitree-trainer"
REMOTE_REPO = "/home/dev/unitree_rl_mjlab"


def sessions(plan: Path) -> list[str]:
  data = json.loads(plan.read_text())
  return [f"{data['group']}_{run['name']}" for run in data["runs"]]


def launch(plan: Path) -> None:
  script = Path(__file__).resolve().with_name("remote_sweep.py")
  subprocess.run(["python3", str(script), str(plan)], check=True)


def status(names: list[str]) -> dict[str, str | None]:
  result = {}
  for name in names:
    path = f"{REMOTE_REPO}/.autotune/{name}"
    command = (
      f"if test -f '{path}/DONE'; then "
      f"cat '{path}/exit_code'; else echo RUNNING; fi"
    )
    output = subprocess.run(
      ["ssh", HOST, command], check=True, capture_output=True, text=True
    ).stdout.strip()
    result[name] = None if output == "RUNNING" else output
  return result


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("first", type=Path)
  parser.add_argument("second", type=Path)
  parser.add_argument("--max-wait-hours", type=float, default=16.0)
  parser.add_argument("--poll-seconds", type=int, default=60)
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()
  first, second = args.first.resolve(), args.second.resolve()
  first_names, second_names = sessions(first), sessions(second)
  if len(first_names) != 4 or len(second_names) != 4:
    raise ValueError("Each wave must contain four distinct one-GPU runs.")
  if args.max_wait_hours <= 0 or args.poll_seconds <= 0:
    raise ValueError("Wait limit and polling interval must be positive.")
  if args.dry_run:
    script = Path(__file__).resolve().with_name("remote_sweep.py")
    for plan in (first, second):
      subprocess.run(["python3", str(script), str(plan), "--dry-run"], check=True)
    return

  launch(first)
  deadline = time.monotonic() + args.max_wait_hours * 3600
  while time.monotonic() < deadline:
    state = status(first_names)
    print(f"first wave: {state}", flush=True)
    if all(code is not None for code in state.values()):
      if any(code != "0" for code in state.values()):
        raise RuntimeError("First wave failed; second wave was not launched.")
      break
    time.sleep(args.poll_seconds)
  else:
    raise TimeoutError("First wave did not finish before the bounded wait limit.")

  subprocess.run(
    [str(Path(__file__).resolve().with_name("remote_status.sh"))], check=True
  )
  launch(second)
  print("Second wave launched.", flush=True)


if __name__ == "__main__":
  main()
