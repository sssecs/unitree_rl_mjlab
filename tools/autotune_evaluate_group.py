#!/usr/bin/env python3
"""Materialize fixed held-out evaluations before a group analysis turn."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(command: list[str], *, cwd: Path, capture: bool = False) -> str:
  result = subprocess.run(
    command, cwd=cwd, check=True, text=True, capture_output=capture
  )
  return result.stdout.strip() if capture else ""


def find_summary(project_root: Path, evaluation_name: str) -> str:
  root = project_root / "results" / "remote_eval" / evaluation_name
  summaries = list(root.rglob("summary.json"))
  manifests = list(root.rglob("manifest.json"))
  episodes = list(root.rglob("episodes.csv"))
  if not (len(summaries) == len(manifests) == len(episodes) == 1):
    raise RuntimeError(f"Incomplete evaluation artifacts under {root}")
  return str(summaries[0].relative_to(project_root))


def write_record(path: Path, record: dict) -> None:
  temporary = path.with_suffix(".json.tmp")
  temporary.write_text(json.dumps(record, indent=2) + "\n")
  temporary.replace(path)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("group_record", type=Path)
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()
  project_root = Path(__file__).resolve().parents[1]
  record = json.loads(args.group_record.read_text())
  plan = json.loads(Path(record["plan_file"]).read_text())
  evaluation = plan.get("evaluation")
  if not evaluation:
    print("Group has no pre-analysis evaluation plan.")
    return

  suites = evaluation.get("suites", ["nominal", "robust"])
  device = evaluation.get("device", "cuda:0")
  checkpoint_name = evaluation.get("checkpoint_name", "model_4999.pt")
  targets: list[tuple[str, str]] = []
  for session in record["sessions"]:
    metadata_text = run(
      [
        "ssh",
        "unitree-trainer",
        f"cat /home/dev/unitree_rl_mjlab/.autotune/{session}/run_metadata.json",
      ],
      cwd=project_root,
      capture=True,
    )
    metadata = json.loads(metadata_text)
    targets.append((session, f"{metadata['log_dir']}/{checkpoint_name}"))

  baseline = evaluation.get("baseline")
  if baseline:
    targets.append((baseline["name"], baseline["checkpoint"]))

  results = record.setdefault("evaluations", {})
  for target_name, checkpoint in targets:
    target_results = results.setdefault(target_name, {})
    for suite in suites:
      eval_name = f"{record['group']}_{target_name}_{suite}_fixed_v1"
      if args.dry_run:
        print(f"{eval_name}: {checkpoint} --suite {suite} --device {device}")
        continue
      run(
        [
          str(project_root / "tools" / "remote_evaluate_wrist.sh"),
          eval_name,
          checkpoint,
          "--suite",
          suite,
          "--device",
          device,
        ],
        cwd=project_root,
      )
      target_results[suite] = find_summary(project_root, eval_name)
      write_record(args.group_record, record)

  if args.dry_run:
    return
  record["evaluation_complete"] = True
  write_record(args.group_record, record)
  print(f"Completed fixed evaluation matrix for {record['group']}")


if __name__ == "__main__":
  main()
