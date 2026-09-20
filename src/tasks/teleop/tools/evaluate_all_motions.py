"""Evaluate one trained teleop checkpoint once on every NPZ motion.

Example:

  python -m src.tasks.teleop.tools.evaluate_all_motions \
    --checkpoint-file logs/rsl_rl/g1_teleop_teacher/.../model_10000.pt \
    --command-dir /data/g1_sparse_npz \
    --batch-size 256 \
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
import time
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
  "left_wrist_lin_vel_error",
  "right_wrist_lin_vel_error",
  "shoulder_mid_xy_error",
  "shoulder_heading_error",
  "left_shoulder_height_error",
  "right_shoulder_height_error",
)


POSTURE_NAMES = (
  "torso_tilt",
  "shoulder_height_delta_error",
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
  parser.add_argument("--batch-size", type=int, default=256)
  parser.add_argument(
    "--progress-interval-steps",
    type=int,
    default=250,
    help=(
      "Synchronize to CPU and print progress every N environment steps. "
      "Larger values reduce CUDA synchronization overhead."
    ),
  )
  parser.add_argument(
    "--preserve-motion-order",
    action="store_true",
    help=(
      "Evaluate batches in motion-id order. By default motions are sorted by "
      "duration before batching to reduce padding/wasted simulation."
    ),
  )
  parser.add_argument("--device", default=None)
  parser.add_argument("--output-dir", default="teleop_eval")
  parser.add_argument("--warmup-duration-s", type=float, default=None)
  parser.add_argument("--recovery-duration-s", type=float, default=None)
  parser.add_argument("--recovery-hold-s", type=float, default=None)
  args = parser.parse_args()

  if args.batch_size <= 0:
    raise ValueError("--batch-size must be positive")
  if args.progress_interval_steps <= 0:
    raise ValueError("--progress-interval-steps must be positive")

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
  mean_duration = float(np.mean(motion_durations))
  median_duration = float(np.median(motion_durations))
  p95_duration = float(np.percentile(motion_durations, 95.0))

  batch_size = min(args.batch_size, num_motions)

  if args.preserve_motion_order:
    eval_order = list(range(num_motions))
  else:
    # Similar-duration clips in the same batch drastically reduce the amount of
    # padded simulation after shorter clips have already terminated.
    eval_order = sorted(
      range(num_motions),
      key=lambda motion_id: float(motion_durations[motion_id]),
      reverse=True,
    )

  batches = [
    eval_order[i : i + batch_size]
    for i in range(0, num_motions, batch_size)
  ]

  # Estimate the padding cost of this batching plan.  This is measured in
  # simulated environment-seconds; ratio=1 is ideal.
  useful_env_seconds = float(np.sum(motion_durations))
  padded_env_seconds = 0.0
  for ids in batches:
    if ids:
      padded_env_seconds += (
        len(ids) * float(np.max(motion_durations[np.asarray(ids)]))
      )
  padding_ratio = (
    padded_env_seconds / useful_env_seconds
    if useful_env_seconds > 0.0 else 1.0
  )

  first = batches[0]
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
    f"batches={len(batches)}, device={device}, checkpoint={checkpoint.name}",
    flush=True,
  )
  print(
    "[eval] motion duration [s]: "
    f"mean={mean_duration:.1f}, median={median_duration:.1f}, "
    f"p95={p95_duration:.1f}, max={max_duration:.1f}",
    flush=True,
  )
  print(
    f"[eval] duration-batching padding ratio={padding_ratio:.2f}x "
    f"(1.00x is ideal)",
    flush=True,
  )
  print(
    "[eval] NOTE: per-step CUDA->CPU synchronization is disabled; "
    f"progress sync occurs every {args.progress_interval_steps} env steps.",
    flush=True,
  )

  eval_wall_start = time.perf_counter()

  for batch_index, real_ids in enumerate(batches):
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
    posture_sum = {
      name: torch.zeros(batch_size, device=device) for name in POSTURE_NAMES
    }
    posture_max = {
      name: torch.zeros(batch_size, device=device) for name in POSTURE_NAMES
    }
    last_posture = {
      name: torch.zeros(batch_size, device=device) for name in POSTURE_NAMES
    }
    max_completion = torch.zeros(batch_size, device=device)
    max_recovery_alpha = torch.zeros(batch_size, device=device)
    reward_sum = torch.zeros(batch_size, device=device)

    batch_max_duration = float(
      np.max(motion_durations[np.asarray(real_ids, dtype=np.int64)])
    )
    max_steps = int(
      math.ceil(
        (
          batch_max_duration
          + float(command.cfg.warmup_duration_s)
          + float(command.cfg.recovery_duration_s)
          + float(command.cfg.recovery_hold_s)
          + 5.0
        )
        / step_dt
      )
    ) + 5

    # Termination/result tensors stay on GPU throughout the rollout.
    result_fell = torch.zeros(batch_size, dtype=torch.bool, device=device)
    result_command_finished = torch.zeros(
      batch_size, dtype=torch.bool, device=device
    )
    result_generic_timeout = torch.zeros(
      batch_size, dtype=torch.bool, device=device
    )

    # Cache term availability once; do not do string/list logic every step.
    tm = env.unwrapped.termination_manager
    has_fell = "fell_over" in tm.active_terms
    has_command_finished = "command_finished" in tm.active_terms
    has_time_out = "time_out" in tm.active_terms

    print(
      f"[eval] batch {batch_index + 1}/{len(batches)}: "
      f"motions={valid_count}, max_duration={batch_max_duration:.1f}s",
      flush=True,
    )

    for step_index in range(max_steps):
      active_env = alive & (~done_once)

      # Stack all tracking errors into one tensor.  This avoids a Python-side
      # loop launching several masked-index kernels per metric.
      error_dict = command.current_tracking_errors()
      error_mat = torch.stack(
        [error_dict[name] for name in ERROR_NAMES],
        dim=0,
      )
      active_f = active_env.to(error_mat.dtype)
      track_mask = active_env & command.motion_tracking_active
      track_f = track_mask.to(error_mat.dtype)

      tracking_steps += track_mask.to(torch.long)
      episode_steps += active_env.to(torch.long)

      error_sum_mat = torch.stack(
        [error_sum[name] for name in ERROR_NAMES], dim=0
      )
      error_max_mat = torch.stack(
        [error_max[name] for name in ERROR_NAMES], dim=0
      )
      last_error_mat = torch.stack(
        [last_error[name] for name in ERROR_NAMES], dim=0
      )

      error_sum_mat += error_mat * track_f.unsqueeze(0)
      error_max_mat = torch.where(
        track_mask.unsqueeze(0),
        torch.maximum(error_max_mat, error_mat),
        error_max_mat,
      )
      last_error_mat = torch.where(
        active_env.unsqueeze(0),
        error_mat,
        last_error_mat,
      )

      # Keep the existing dict buffers backed by the updated tensors.
      for metric_index, name in enumerate(ERROR_NAMES):
        error_sum[name] = error_sum_mat[metric_index]
        error_max[name] = error_max_mat[metric_index]
        last_error[name] = last_error_mat[metric_index]

      # V7 posture diagnostics are evaluated over the same recorded-motion
      # phase as task errors, but never contribute to reward.
      posture_dict = command.current_posture_diagnostics()
      posture_mat = torch.stack(
        [posture_dict[name] for name in POSTURE_NAMES],
        dim=0,
      )
      posture_sum_mat = torch.stack(
        [posture_sum[name] for name in POSTURE_NAMES], dim=0
      )
      posture_max_mat = torch.stack(
        [posture_max[name] for name in POSTURE_NAMES], dim=0
      )
      last_posture_mat = torch.stack(
        [last_posture[name] for name in POSTURE_NAMES], dim=0
      )
      posture_sum_mat += posture_mat * track_f.unsqueeze(0)
      posture_max_mat = torch.where(
        track_mask.unsqueeze(0),
        torch.maximum(posture_max_mat, posture_mat),
        posture_max_mat,
      )
      last_posture_mat = torch.where(
        active_env.unsqueeze(0),
        posture_mat,
        last_posture_mat,
      )
      for metric_index, name in enumerate(POSTURE_NAMES):
        posture_sum[name] = posture_sum_mat[metric_index]
        posture_max[name] = posture_max_mat[metric_index]
        last_posture[name] = last_posture_mat[metric_index]

      max_completion = torch.maximum(
        max_completion,
        command.motion_completion_ratio * active_f,
      )
      max_recovery_alpha = torch.maximum(
        max_recovery_alpha,
        command.recovery_alpha * active_f,
      )

      with torch.inference_mode():
        actions = policy(obs)
      obs, reward, dones, _extras = env.step(actions)
      reward_sum += reward * active_f

      dones_bool = dones.to(torch.bool)
      newly_done = active_env & dones_bool

      if has_fell:
        result_fell |= newly_done & tm.get_term("fell_over")
      if has_command_finished:
        result_command_finished |= (
          newly_done & tm.get_term("command_finished")
        )
      if has_time_out:
        result_generic_timeout |= newly_done & tm.get_term("time_out")

      done_once |= newly_done

      # IMPORTANT: this is the only periodic GPU->CPU synchronization in the
      # rollout loop.  The previous evaluator synchronized at least twice EVERY
      # step through Python `if torch.any(cuda_tensor)` branches.
      if (
        (step_index + 1) % args.progress_interval_steps == 0
        or step_index + 1 == max_steps
      ):
        completed_now = int(
          done_once[:valid_count].sum().detach().cpu().item()
        )
        sim_time = (step_index + 1) * step_dt
        print(
          f"[eval] batch {batch_index + 1}/{len(batches)} "
          f"sim_t={sim_time:.1f}s "
          f"completed={completed_now}/{valid_count}",
          flush=True,
        )
        if completed_now >= valid_count:
          break

    # Transfer one compact snapshot to CPU ONCE per batch.
    tracking_steps_cpu = tracking_steps[:valid_count].detach().cpu().numpy()
    episode_steps_cpu = episode_steps[:valid_count].detach().cpu().numpy()
    max_completion_cpu = max_completion[:valid_count].detach().cpu().numpy()
    reward_sum_cpu = reward_sum[:valid_count].detach().cpu().numpy()
    fell_cpu = result_fell[:valid_count].detach().cpu().numpy()
    finished_cpu = result_command_finished[:valid_count].detach().cpu().numpy()
    timeout_cpu = result_generic_timeout[:valid_count].detach().cpu().numpy()
    done_cpu = done_once[:valid_count].detach().cpu().numpy()

    error_sum_cpu = {
      name: error_sum[name][:valid_count].detach().cpu().numpy()
      for name in ERROR_NAMES
    }
    error_max_cpu = {
      name: error_max[name][:valid_count].detach().cpu().numpy()
      for name in ERROR_NAMES
    }
    last_error_cpu = {
      name: last_error[name][:valid_count].detach().cpu().numpy()
      for name in ERROR_NAMES
    }
    posture_sum_cpu = {
      name: posture_sum[name][:valid_count].detach().cpu().numpy()
      for name in POSTURE_NAMES
    }
    posture_max_cpu = {
      name: posture_max[name][:valid_count].detach().cpu().numpy()
      for name in POSTURE_NAMES
    }
    last_posture_cpu = {
      name: last_posture[name][:valid_count].detach().cpu().numpy()
      for name in POSTURE_NAMES
    }

    for local_idx in range(valid_count):
      motion_id = padded_ids[local_idx]
      n_track = int(tracking_steps_cpu[local_idx])
      row = {
        "motion_id": motion_id,
        "motion_name": motion_names[motion_id],
        "motion_duration_s": float(motion_durations[motion_id]),
        "episode_steps": int(episode_steps_cpu[local_idx]),
        "tracking_steps": n_track,
        "motion_completion_ratio": float(max_completion_cpu[local_idx]),
        "motion_completed": bool(
          max_completion_cpu[local_idx] >= 1.0 - 1e-5
        ),
        "recovery_completed": bool(finished_cpu[local_idx]),
        "fell_over": bool(fell_cpu[local_idx]),
        "generic_time_out": bool(
          timeout_cpu[local_idx] or (not done_cpu[local_idx])
        ),
        "episode_reward_sum": float(reward_sum_cpu[local_idx]),
      }
      for name in ERROR_NAMES:
        if n_track > 0:
          row[f"mean_{name}"] = float(
            error_sum_cpu[name][local_idx] / n_track
          )
          row[f"max_{name}"] = float(
            error_max_cpu[name][local_idx]
          )
        else:
          row[f"mean_{name}"] = float("nan")
          row[f"max_{name}"] = float("nan")
        row[f"final_{name}"] = float(last_error_cpu[name][local_idx])

      for name in POSTURE_NAMES:
        if n_track > 0:
          row[f"mean_{name}"] = float(
            posture_sum_cpu[name][local_idx] / n_track
          )
          row[f"max_{name}"] = float(
            posture_max_cpu[name][local_idx]
          )
        else:
          row[f"mean_{name}"] = float("nan")
          row[f"max_{name}"] = float("nan")
        row[f"final_{name}"] = float(last_posture_cpu[name][local_idx])

      rows.append(row)

    print(
      f"[eval] completed {len(rows)}/{num_motions}",
      flush=True,
    )

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
    "posture_diagnostics": {
      "mean": {
        name: mean_field(f"mean_{name}") for name in POSTURE_NAMES
      },
      "mean_of_clip_max": {
        name: mean_field(f"max_{name}") for name in POSTURE_NAMES
      },
      "mean_final": {
        name: mean_field(f"final_{name}") for name in POSTURE_NAMES
      },
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

  eval_wall_s = time.perf_counter() - eval_wall_start
  print(f"[eval] wrote {csv_path}")
  print(f"[eval] wrote {json_path}")
  print(
    f"[eval] evaluation wall time={eval_wall_s:.1f}s "
    f"({eval_wall_s / max(num_motions, 1):.3f}s/motion)",
    flush=True,
  )
  print(
    "[eval] completion={:.1%}, recovery={:.1%}, fall={:.1%}".format(
      summary["motion_completion_rate"],
      summary["recovery_completion_rate"],
      summary["fall_rate"],
    )
  )


if __name__ == "__main__":
  main()
