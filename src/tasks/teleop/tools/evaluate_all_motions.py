"""Evaluate one trained teleop checkpoint once on every NPZ motion.

Example:

  python -m src.tasks.teleop.tools.evaluate_all_motions \
    --checkpoint-file logs/rsl_rl/g1_teleop_teacher/.../model_10000.pt \
    --command-dir /data/g1_sparse_npz \
    --batch-size 32 \
    --output-dir eval/model_10000

The evaluation is deterministic with respect to environment reset placement and
physics randomization: push/encoder/friction/COM randomization and actor
observation corruption are disabled.  Every motion is run from t=0, including
warm-up and the configured return-to-neutral recovery phase.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from src.tasks.teleop.mdp.commands import SparseWholeBodyCommand
from src.tasks.teleop.mdp.motion_library import SparseWholeBodyMotionLibrary


ERROR_NAMES = (
  "left_wrist_pos_error",
  "right_wrist_pos_error",
  "left_wrist_ori_error",
  "right_wrist_ori_error",
  "shoulder_mid_xy_error",
  "shoulder_heading_error",
  "left_shoulder_height_error",
  "right_shoulder_height_error",
)


def _pad_batch(ids: list[int], batch_size: int) -> list[int]:
  if not ids:
    raise ValueError("Cannot pad an empty motion batch")
  return ids + [ids[-1]] * (batch_size - len(ids))


def _json_safe(value):
  if isinstance(value, (np.floating, float)):
    value = float(value)
    return None if not math.isfinite(value) else value
  if isinstance(value, (np.integer, int)):
    return int(value)
  if isinstance(value, (np.bool_, bool)):
    return bool(value)
  if isinstance(value, dict):
    return {str(k): _json_safe(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(v) for v in value]
  return value


def _configure_eval_env(
  task_id: str,
  command_dir: str,
  fixed_motion_ids: tuple[int, ...],
  batch_size: int,
  max_motion_duration_s: float,
  warmup_duration_s: float | None,
  recovery_duration_s: float | None,
  recovery_hold_s: float | None,
):
  # Use the TRAIN cfg, then explicitly make it deterministic.  This preserves
  # command_finished, unlike the interactive play cfg.
  env_cfg = load_env_cfg(task_id, play=False)
  env_cfg.scene.num_envs = batch_size
  env_cfg.observations["actor"].enable_corruption = False

  # No stochastic perturbation in benchmark evaluation.
  for name in ("push_robot", "foot_friction", "encoder_bias", "base_com"):
    env_cfg.events.pop(name, None)

  reset_pose = env_cfg.events["reset_base"].params["pose_range"]
  reset_pose["x"] = (0.0, 0.0)
  reset_pose["y"] = (0.0, 0.0)
  reset_pose["yaw"] = (0.0, 0.0)

  cmd_cfg = env_cfg.commands["teleop"]
  cmd_cfg.command_dir = command_dir
  cmd_cfg.command_file = ""
  cmd_cfg.fixed_motion_id = None
  cmd_cfg.fixed_motion_ids = fixed_motion_ids
  cmd_cfg.sampling_mode = "start"
  cmd_cfg.loop = False
  cmd_cfg.post_motion_behavior = "recover"
  cmd_cfg.debug_vis = False

  if warmup_duration_s is not None:
    cmd_cfg.warmup_duration_s = warmup_duration_s
  if recovery_duration_s is not None:
    cmd_cfg.recovery_duration_s = recovery_duration_s
  if recovery_hold_s is not None:
    cmd_cfg.recovery_hold_s = recovery_hold_s

  # Ensure the generic time_out cannot preempt the longest evaluated motion.
  env_cfg.episode_length_s = (
    max_motion_duration_s
    + float(cmd_cfg.warmup_duration_s)
    + float(cmd_cfg.recovery_duration_s)
    + float(cmd_cfg.recovery_hold_s)
    + 5.0
  )
  return env_cfg


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint-file", required=True)
  parser.add_argument("--command-dir", required=True)
  parser.add_argument("--task-id", default="Unitree-G1-Teleop")
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--device", default=None)
  parser.add_argument("--output-dir", default="teleop_eval")
  parser.add_argument("--warmup-duration-s", type=float, default=None)
  parser.add_argument("--recovery-duration-s", type=float, default=None)
  parser.add_argument("--recovery-hold-s", type=float, default=None)
  args = parser.parse_args()

  if args.batch_size <= 0:
    raise ValueError("--batch-size must be positive")

  checkpoint = Path(args.checkpoint_file).expanduser().resolve()
  command_dir = Path(args.command_dir).expanduser().resolve()
  output_dir = Path(args.output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  if not command_dir.is_dir():
    raise NotADirectoryError(command_dir)

  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Import task package for registry side effects.
  import src.tasks  # noqa: F401

  # CPU enumeration guarantees evaluation uses exactly the same sorted/valid
  # motion-id mapping as the training command library.
  base_env_cfg = load_env_cfg(args.task_id, play=False)
  base_cmd_cfg = base_env_cfg.commands["teleop"]
  catalog = SparseWholeBodyMotionLibrary(
    command_source=str(command_dir),
    device="cpu",
    canonicalize_heading=base_cmd_cfg.canonicalize_heading,
    recursive=base_cmd_cfg.recursive_scan,
    skip_invalid_files=base_cmd_cfg.skip_invalid_files,
    sampling_weight_mode=base_cmd_cfg.motion_sampling_weight_mode,
  )
  num_motions = catalog.num_motions
  motion_names = list(catalog.motion_names)
  motion_durations = catalog.motion_lengths.cpu().numpy()
  max_duration = float(np.max(motion_durations))

  batch_size = min(args.batch_size, num_motions)
  first = list(range(min(batch_size, num_motions)))
  first_padded = tuple(_pad_batch(first, batch_size))

  env_cfg = _configure_eval_env(
    args.task_id,
    str(command_dir),
    first_padded,
    batch_size,
    max_duration,
    args.warmup_duration_s,
    args.recovery_duration_s,
    args.recovery_hold_s,
  )
  agent_cfg = load_rl_cfg(args.task_id)

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)

  rows: list[dict] = []
  step_dt = float(env.unwrapped.step_dt)
  command = env.unwrapped.command_manager.get_term("teleop")
  assert isinstance(command, SparseWholeBodyCommand)

  print(
    f"[eval] motions={num_motions}, batch_size={batch_size}, "
    f"device={device}, checkpoint={checkpoint.name}"
  )

  for batch_begin in range(0, num_motions, batch_size):
    real_ids = list(range(batch_begin, min(batch_begin + batch_size, num_motions)))
    padded_ids = _pad_batch(real_ids, batch_size)
    valid_count = len(real_ids)

    command.cfg.fixed_motion_ids = tuple(padded_ids)
    command.cfg.fixed_motion_id = None
    obs, _ = env.reset()

    alive = torch.zeros(batch_size, dtype=torch.bool, device=device)
    alive[:valid_count] = True
    done_once = ~alive

    tracking_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
    episode_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
    error_sum = {
      name: torch.zeros(batch_size, device=device) for name in ERROR_NAMES
    }
    error_max = {
      name: torch.zeros(batch_size, device=device) for name in ERROR_NAMES
    }
    last_error = {
      name: torch.zeros(batch_size, device=device) for name in ERROR_NAMES
    }
    max_completion = torch.zeros(batch_size, device=device)
    max_recovery_alpha = torch.zeros(batch_size, device=device)
    reward_sum = torch.zeros(batch_size, device=device)

    max_steps = int(math.ceil(env_cfg.episode_length_s / step_dt)) + 5

    for _ in range(max_steps):
      active_env = alive & (~done_once)
      if not torch.any(active_env):
        break

      errors = command.current_tracking_errors()
      track_mask = active_env & command.motion_tracking_active
      tracking_steps += track_mask.to(torch.long)
      episode_steps += active_env.to(torch.long)

      for name, value in errors.items():
        last_error[name][active_env] = value[active_env]
        error_sum[name] += value * track_mask.to(value.dtype)
        error_max[name] = torch.where(
          track_mask,
          torch.maximum(error_max[name], value),
          error_max[name],
        )

      max_completion = torch.maximum(
        max_completion,
        torch.where(
          active_env,
          command.motion_completion_ratio,
          torch.zeros_like(max_completion),
        ),
      )
      max_recovery_alpha = torch.maximum(
        max_recovery_alpha,
        torch.where(
          active_env,
          command.recovery_alpha,
          torch.zeros_like(max_recovery_alpha),
        ),
      )

      with torch.inference_mode():
        actions = policy(obs)
      obs, reward, dones, _extras = env.step(actions)
      reward_sum += reward * active_env.to(reward.dtype)

      dones_bool = dones.to(torch.bool)
      newly_done = active_env & dones_bool
      if not torch.any(newly_done):
        continue

      tm = env.unwrapped.termination_manager
      fell = tm.get_term("fell_over") if "fell_over" in tm.active_terms else torch.zeros_like(newly_done)
      command_finished = (
        tm.get_term("command_finished")
        if "command_finished" in tm.active_terms
        else torch.zeros_like(newly_done)
      )
      generic_timeout = (
        tm.get_term("time_out")
        if "time_out" in tm.active_terms
        else torch.zeros_like(newly_done)
      )

      for local_idx in newly_done.nonzero(as_tuple=False).flatten().tolist():
        if local_idx >= valid_count:
          continue
        motion_id = padded_ids[local_idx]
        n_track = int(tracking_steps[local_idx].item())
        row = {
          "motion_id": motion_id,
          "motion_name": motion_names[motion_id],
          "motion_duration_s": float(motion_durations[motion_id]),
          "episode_steps": int(episode_steps[local_idx].item()),
          "tracking_steps": n_track,
          "motion_completion_ratio": float(max_completion[local_idx].item()),
          "motion_completed": bool(max_completion[local_idx].item() >= 1.0 - 1e-5),
          "recovery_completed": bool(command_finished[local_idx].item()),
          "fell_over": bool(fell[local_idx].item()),
          "generic_time_out": bool(generic_timeout[local_idx].item()),
          "episode_reward_sum": float(reward_sum[local_idx].item()),
        }
        for name in ERROR_NAMES:
          if n_track > 0:
            row[f"mean_{name}"] = float(
              (error_sum[name][local_idx] / n_track).item()
            )
            row[f"max_{name}"] = float(error_max[name][local_idx].item())
          else:
            row[f"mean_{name}"] = float("nan")
            row[f"max_{name}"] = float("nan")
          # Error at the final pre-termination observation.  For successful
          # episodes this is the held neutral/recovery target error.
          row[f"final_{name}"] = float(last_error[name][local_idx].item())

        rows.append(row)

      done_once |= newly_done

    # Safety fallback if an environment never terminated.
    for local_idx in range(valid_count):
      if done_once[local_idx]:
        continue
      motion_id = padded_ids[local_idx]
      n_track = int(tracking_steps[local_idx].item())
      row = {
        "motion_id": motion_id,
        "motion_name": motion_names[motion_id],
        "motion_duration_s": float(motion_durations[motion_id]),
        "episode_steps": int(episode_steps[local_idx].item()),
        "tracking_steps": n_track,
        "motion_completion_ratio": float(max_completion[local_idx].item()),
        "motion_completed": bool(max_completion[local_idx].item() >= 1.0 - 1e-5),
        "recovery_completed": False,
        "fell_over": False,
        "generic_time_out": True,
        "episode_reward_sum": float(reward_sum[local_idx].item()),
      }
      for name in ERROR_NAMES:
        row[f"mean_{name}"] = (
          float((error_sum[name][local_idx] / n_track).item())
          if n_track > 0 else float("nan")
        )
        row[f"max_{name}"] = (
          float(error_max[name][local_idx].item())
          if n_track > 0 else float("nan")
        )
        row[f"final_{name}"] = float(last_error[name][local_idx].item())
      rows.append(row)

    print(f"[eval] completed {min(batch_begin + valid_count, num_motions)}/{num_motions}")

  env.close()

  rows.sort(key=lambda r: r["motion_id"])

  # CSV without pandas dependency.
  import csv

  csv_path = output_dir / "per_motion.csv"
  with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

  def mean_field(name: str) -> float:
    values = np.asarray([r[name] for r in rows], dtype=np.float64)
    return float(np.nanmean(values))

  combined_wrist = [
    max(r["mean_left_wrist_pos_error"], r["mean_right_wrist_pos_error"])
    for r in rows
  ]
  worst_order = np.argsort(np.asarray(combined_wrist))[::-1][:20]

  summary = {
    "checkpoint": str(checkpoint),
    "command_dir": str(command_dir),
    "num_motions": num_motions,
    "motion_completion_rate": float(np.mean([r["motion_completed"] for r in rows])),
    "recovery_completion_rate": float(np.mean([r["recovery_completed"] for r in rows])),
    "fall_rate": float(np.mean([r["fell_over"] for r in rows])),
    "mean_motion_completion_ratio": mean_field("motion_completion_ratio"),
    "mean_errors": {
      name: mean_field(f"mean_{name}") for name in ERROR_NAMES
    },
    "mean_max_errors": {
      name: mean_field(f"max_{name}") for name in ERROR_NAMES
    },
    "mean_final_recovery_errors": {
      name: mean_field(f"final_{name}") for name in ERROR_NAMES
    },
    "worst_20_by_mean_wrist_position_error": [
      {
        "motion_id": rows[int(i)]["motion_id"],
        "motion_name": rows[int(i)]["motion_name"],
        "worst_hand_mean_pos_error_m": float(combined_wrist[int(i)]),
        "completion_ratio": rows[int(i)]["motion_completion_ratio"],
        "fell_over": rows[int(i)]["fell_over"],
      }
      for i in worst_order
    ],
  }

  json_path = output_dir / "summary.json"
  json_path.write_text(
    json.dumps(_json_safe(summary), indent=2, ensure_ascii=False),
    encoding="utf-8",
  )

  print(f"[eval] wrote {csv_path}")
  print(f"[eval] wrote {json_path}")
  print(
    "[eval] completion={:.1%}, recovery={:.1%}, fall={:.1%}".format(
      summary["motion_completion_rate"],
      summary["recovery_completion_rate"],
      summary["fall_rate"],
    )
  )


if __name__ == "__main__":
  main()
