"""Deterministic held-out evaluation for the G1 wrist-recovery policy."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_error_magnitude
from mjlab.utils.torch import configure_torch_backends

from src.tasks.wrist_recovery.mdp import BimanualWristCommand


TASK_ID = "Unitree-G1-Wrist-Recovery-Teacher"
EVALUATOR_VERSION = 1
ALL_SCENARIOS = (
  "static_hold",
  "payload",
  "push",
  "symmetric_reach",
  "asymmetric_reach",
  "combined",
)


@dataclass(frozen=True)
class EvalCfg:
  checkpoint_file: str
  output_dir: str = "results/wrist_recovery_eval"
  device: str | None = None
  num_envs: int = 64
  seeds: tuple[int, ...] = (1103, 2207, 3301)
  scenario: Literal[
    "all", "static_hold", "payload", "push", "symmetric_reach",
    "asymmetric_reach", "combined"
  ] = "all"
  suite: Literal["nominal", "robust"] = "nominal"
  horizon_s: float = 12.0


def _scenario_inputs(
  name: str, num_envs: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float | None]:
  """Return target offsets, axis angles, wrist forces, root velocity kicks."""
  offsets = torch.zeros(num_envs, 2, 3, device=device)
  rotations = torch.zeros_like(offsets)
  forces = torch.zeros_like(offsets)
  push = torch.zeros(num_envs, 6, device=device)
  push_time = None
  idx = torch.arange(num_envs, device=device)
  severity = torch.tensor((0.5, 0.75, 1.0), device=device)[idx % 3]
  sign = torch.where((idx // 3) % 2 == 0, 1.0, -1.0)

  if name in ("payload", "combined"):
    forces[..., 2] = -12.0 * severity[:, None]
    # One third of cases are asymmetric loads.
    forces[idx % 3 == 1, 0, 2] *= 0.5
  if name in ("symmetric_reach", "combined"):
    offsets[..., 0] = (0.12 + 0.16 * severity)[:, None]
  if name == "asymmetric_reach":
    offsets[:, 0, 0] = 0.12 + 0.16 * severity
    offsets[:, 1, 0] = 0.08 + 0.10 * severity
    offsets[:, 0, 1] = 0.08 * sign
    offsets[:, 1, 1] = -0.05 * sign
    offsets[:, 0, 2] = 0.05 * sign
    rotations[:, 0, 1] = 0.20 * sign
    rotations[:, 1, 2] = -0.15 * sign
  if name in ("push", "combined"):
    push_time = 4.0 if name == "push" else 5.0
    axes = idx % 3
    push[axes == 0, 0] = 0.6 * severity[axes == 0] * sign[axes == 0]
    push[axes == 1, 1] = 0.6 * severity[axes == 1] * sign[axes == 1]
    push[axes == 2, 5] = 0.3 * severity[axes == 2] * sign[axes == 2]
  return offsets, rotations, forces, push, push_time


def _install_wrench(
  env: ManagerBasedRlEnv, command: BimanualWristCommand, forces: torch.Tensor
) -> None:
  torques = torch.zeros_like(forces)
  command.robot.write_external_wrench_to_sim(
    forces, torques, body_ids=command.wrist_body_ids
  )
  env._teacher_hand_wrench_w = torch.cat((forces, torques), dim=-1)


def _evaluate_scenario(env, policy, name: str, seed: int) -> list[dict]:
  raw = env.unwrapped
  raw.reset(seed=seed)
  command = cast(
    BimanualWristCommand, raw.command_manager.get_term("wrists")
  )
  offsets, rotations, forces, push, push_time = _scenario_inputs(
    name, raw.num_envs, raw.device
  )
  command.set_scripted_trajectory(
    offsets,
    rotations,
    delay_s=1.0,
    duration_s=2.0,
    scenario_code=0.0 if name in ("static_hold", "payload", "push") else 1.0,
  )
  _install_wrench(raw, command, forces)
  # Fill the three-frame history with the installed held-out command/wrench.
  for _ in range(3):
    raw.obs_buf = raw.observation_manager.compute(update_history=True)
  obs = env.get_observations()

  foot0 = command.robot.data.body_link_pos_w[:, command.foot_body_ids].clone()
  active = torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
  fell = torch.zeros_like(active)
  sums = {k: torch.zeros(raw.num_envs, device=raw.device) for k in
          ("wrist_pos", "wrist_rot", "base_speed", "yaw_rate", "foot_slip")}
  peaks = {k: torch.zeros(raw.num_envs, device=raw.device) for k in
           ("wrist_pos", "wrist_rot", "foot_displacement")}
  contact_changes = torch.zeros(raw.num_envs, device=raw.device)
  previous_contact = None
  samples = torch.zeros(raw.num_envs, device=raw.device)
  push_step = None if push_time is None else round(push_time / raw.step_dt)
  total_steps = round(raw.cfg.episode_length_s / raw.step_dt) - 1

  # MuJoCo-Warp stores some step outputs in persistent buffers; inference-mode
  # tensors cannot later be reset in-place between scenarios.
  with torch.no_grad():
    for step in range(total_steps):
      if push_step is not None and step == push_step:
        velocity = command.robot.data.root_link_vel_w.clone() + push
        command.robot.write_root_link_velocity_to_sim(velocity)
        raw._teacher_push_delta_w = push.clone()
        raw.scene.write_data_to_sim()
        raw.sim.forward()
        raw.sim.sense()
        raw.obs_buf = raw.observation_manager.compute(update_history=True)
        obs = env.get_observations()

      obs, _, dones, _ = env.step(policy(obs))
      newly_done = dones.bool() & active
      # The vector environment has already reset terminated instances. Exclude
      # that post-reset sample from trajectory errors and peak statistics.
      valid_mask = active & ~newly_done
      valid = valid_mask.float()
      pos = torch.linalg.norm(
        command.desired_wrist_pos_w - command.robot_wrist_pos_w, dim=-1
      ).mean(-1)
      rot = quat_error_magnitude(
        command.desired_wrist_quat_w, command.robot_wrist_quat_w
      ).mean(-1)
      root_vel = command.robot.data.root_link_lin_vel_w
      speed = torch.linalg.norm(root_vel[:, :2], dim=-1)
      yaw = command.robot.data.root_link_ang_vel_w[:, 2].abs()
      feet = command.robot.data.body_link_pos_w[:, command.foot_body_ids]
      foot_disp = torch.linalg.norm(
        feet[..., :2] - foot0[..., :2], dim=-1
      ).max(-1).values
      foot_speed = torch.linalg.norm(
        command.robot.data.body_link_lin_vel_w[:, command.foot_body_ids, :2], dim=-1
      )
      contact = raw.scene["feet_ground_contact"].data.found > 0
      slip = (foot_speed * contact.float()).mean(-1)
      if previous_contact is not None:
        contact_changes += (
          (contact != previous_contact).any(-1) & valid_mask
        ).float()
      previous_contact = contact
      for key, value in (("wrist_pos", pos), ("wrist_rot", rot),
                         ("base_speed", speed), ("yaw_rate", yaw),
                         ("foot_slip", slip)):
        sums[key] += value * valid
      for key, value in (("wrist_pos", pos), ("wrist_rot", rot),
                         ("foot_displacement", foot_disp)):
        peaks[key] = torch.maximum(peaks[key], value * valid)
      samples += valid
      fell |= newly_done
      active &= ~newly_done

  denom = samples.clamp_min(1.0)
  final_pos = torch.linalg.norm(
    command.desired_wrist_pos_w - command.robot_wrist_pos_w, dim=-1
  ).mean(-1)
  final_rot = quat_error_magnitude(
    command.desired_wrist_quat_w, command.robot_wrist_quat_w
  ).mean(-1)
  rows = []
  for i in range(raw.num_envs):
    rows.append({
      "scenario": name, "seed": seed, "env": i,
      "fell": bool(fell[i].item()),
      "success": bool(
        (~fell[i] & (final_pos[i] < 0.04) & (final_rot[i] < 0.20)).item()
      ),
      "wrist_pos_mean_m": float((sums["wrist_pos"][i] / denom[i]).item()),
      "wrist_pos_peak_m": float(peaks["wrist_pos"][i].item()),
      "wrist_rot_mean_rad": float((sums["wrist_rot"][i] / denom[i]).item()),
      "wrist_rot_peak_rad": float(peaks["wrist_rot"][i].item()),
      "base_speed_mean_mps": float((sums["base_speed"][i] / denom[i]).item()),
      "yaw_rate_mean_rps": float((sums["yaw_rate"][i] / denom[i]).item()),
      "foot_slip_mean_mps": float((sums["foot_slip"][i] / denom[i]).item()),
      "foot_displacement_peak_m": float(peaks["foot_displacement"][i].item()),
      "contact_changes": float(contact_changes[i].item()),
    })
  return rows


def _summarize(rows: list[dict]) -> dict:
  result = {}
  for scenario in ALL_SCENARIOS:
    subset = [r for r in rows if r["scenario"] == scenario]
    if not subset:
      continue
    summary = {"episodes": len(subset)}
    for key in ("success", "fell"):
      summary[f"{key}_rate"] = sum(float(r[key]) for r in subset) / len(subset)
    for key in (
      "wrist_pos_mean_m", "wrist_pos_peak_m", "wrist_rot_mean_rad",
      "wrist_rot_peak_rad", "base_speed_mean_mps", "yaw_rate_mean_rps",
      "foot_slip_mean_mps", "foot_displacement_peak_m", "contact_changes",
    ):
      values = torch.tensor([r[key] for r in subset])
      summary[key] = {
        "mean": float(values.mean()),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
      }
    result[scenario] = summary
  return result


def main() -> None:
  cfg = tyro.cli(EvalCfg)
  if not cfg.seeds:
    raise ValueError("At least one evaluation seed is required.")
  configure_torch_backends()
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  checkpoint = Path(cfg.checkpoint_file).resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(TASK_ID, play=False)
  agent_cfg = load_rl_cfg(TASK_ID)
  env_cfg.seed = cfg.seeds[0]
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.episode_length_s = (
    cfg.horizon_s + env_cfg.sim.mujoco.timestep * env_cfg.decimation
  )
  env_cfg.events.pop("push_robot", None)
  env_cfg.events.pop("hand_payload", None)
  if cfg.suite == "nominal":
    for event in ("foot_friction", "encoder_bias", "base_com"):
      env_cfg.events.pop(event, None)
  for group in env_cfg.observations.values():
    group.enable_corruption = False

  raw = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  scenarios = ALL_SCENARIOS if cfg.scenario == "all" else (cfg.scenario,)
  rows = []
  for seed in cfg.seeds:
    for scenario in scenarios:
      print(f"[eval] seed={seed} scenario={scenario}")
      rows.extend(_evaluate_scenario(env, policy, scenario, seed))
  env.close()

  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  run_dir = Path(cfg.output_dir) / f"{stamp}_{checkpoint.stem}_{cfg.suite}"
  run_dir.mkdir(parents=True, exist_ok=False)
  fields = list(rows[0])
  with (run_dir / "episodes.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
  summary = _summarize(rows)
  (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
  manifest = {
    **asdict(cfg),
    "checkpoint_file": str(checkpoint),
    "task": TASK_ID,
    "evaluator_version": EVALUATOR_VERSION,
  }
  (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  print(json.dumps(summary, indent=2))
  print(f"[eval] wrote {run_dir}")


if __name__ == "__main__":
  main()
