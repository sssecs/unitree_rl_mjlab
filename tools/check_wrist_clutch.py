#!/usr/bin/env python3
"""CPU checks of actual clutch methods, without loading simulator libraries."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
from collections import defaultdict
import torch
from check_wrist_command_sampling import fixture
from summarize_wrist_training import conditional_metrics

ROOT = Path(__file__).resolve().parents[1]


def methods(path, cls, names):
  tree = ast.parse((ROOT / path).read_text())
  node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
  return [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names]


def apply(q, v):
  xyz = q[..., 1:]
  cross = 2 * torch.linalg.cross(xyz, v)
  return v + q[..., :1] * cross + torch.linalg.cross(xyz, cross)


def mul(a, b):
  return torch.cat((a[..., :1]*b[..., :1] - (a[..., 1:]*b[..., 1:]).sum(-1, keepdim=True),
    a[..., :1]*b[..., 1:] + b[..., :1]*a[..., 1:] + torch.linalg.cross(a[..., 1:], b[..., 1:])), -1)


def angle_axis(angle, axis):
  return torch.cat((torch.cos(angle/2)[:, None], axis*torch.sin(angle/2)[:, None]), -1)


def main():
  scope = {"torch": torch, "quat_apply": apply, "quat_mul": mul, "quat_from_angle_axis": angle_axis}
  nodes = methods("src/tasks/wrist_recovery/mdp/clutch.py", "ClutchedVelocityCommand",
                  ("_resample_command", "_update_command"))
  nodes += methods("src/tasks/wrist_recovery/mdp/commands.py", "BimanualWristCommand",
                   ("_advance_transport_reference", "_update_metrics"))
  scope["quat_error_magnitude"] = lambda a,b: torch.zeros(a.shape[:-1])
  exec(compile(ast.Module(body=nodes, type_ignores=[]), "clutch-methods", "exec"), scope)
  cfg = NS(clutch_enabled=True, transport_probability=.25, adjust_probability=.25,
    curriculum_warmup_steps=30000, curriculum_ramp_steps=60000,
    ranges=NS(lin_vel_x=(-.1,.1), lin_vel_y=(-.05,.05), ang_vel_z=(-.1,.1)),
    adjust_linear_speed=.04, adjust_angular_speed=.06, adjust_distance_limit=.08,
    adjust_yaw_limit=.12, adjust_delay_s=3., adjust_duration_s=2.)
  n = 1024
  x = NS(cfg=cfg, device="cpu", _env=NS(common_step_counter=90000, step_dt=.02))
  x.mode = torch.zeros(n, dtype=torch.long)
  x.is_standing_env = torch.zeros(n, dtype=torch.bool)
  x.sampled_twist = torch.zeros(n, 3)
  x.vel_command_b = torch.zeros(n, 3)
  for name in ("adjust_distance", "adjust_yaw", "age"):
    setattr(x, name, torch.zeros(n))
  ids = torch.arange(0,n,2).flip(0)
  torch.manual_seed(1802)
  scope["_resample_command"](x, ids)
  assert torch.count_nonzero(x.sampled_twist[1::2]) == 0
  assert all((x.mode[ids] == i).sum() > 60 for i in range(3))
  for _ in range(700):
    scope["_update_command"](x)
    assert torch.count_nonzero(x.vel_command_b[x.mode == 0]) == 0
    assert x.adjust_distance.max() <= .080001
    assert x.adjust_yaw.max() <= .120001
  assert torch.count_nonzero(x.vel_command_b[x.mode == 2]) == 0
  assert x.adjust_distance.max() > .03
  print("PASS three modes, noncontiguous reset, bounded adjustment and stop")
  # A transport sample translates/rotates around the independent reference;
  # anchored modes remain bitwise unchanged, regardless of actual robot drift.
  twist = NS(cfg=cfg, mode=torch.tensor([1,2,0]), command=torch.tensor([[.1,0.,.1]]).repeat(3,1))
  w = NS(_env=NS(step_dt=1., command_manager=NS(get_term=lambda _:twist)),
    transport_reference_pos=torch.zeros(3,3), transport_reference_yaw=torch.zeros(3))
  for key in ("start_pos_w", "final_pos_w"):
    setattr(w,key,torch.tensor([[[1.,0.,.5],[1.,.2,.5]]]).repeat(3,1,1))
  for key in ("start_quat_w", "final_quat_w"):
    setattr(w,key,torch.tensor([1.,0.,0.,0.]).repeat(3,2,1))
  before = w.start_pos_w.clone()
  scope["_advance_transport_reference"](w)
  assert torch.equal(w.start_pos_w[1:], before[1:])
  assert torch.allclose(w.start_pos_w[0,0], torch.tensor([torch.cos(torch.tensor(.1))+.1, torch.sin(torch.tensor(.1)), .5]))
  assert torch.allclose(w.start_pos_w[...,2], before[...,2])
  assert torch.allclose(torch.linalg.vector_norm(w.start_quat_w,dim=-1),torch.ones(3,2))
  scope["_advance_transport_reference"](w)
  assert torch.equal(w.start_pos_w[1:], before[1:])
  print("PASS independent commanded translation/yaw, world-fixed anchored wrists, unchanged height")
  # A terminal snapshot after the adjustment stops must retain the moving
  # interval's errors, rather than reporting a zero/unavailable moving fraction.
  z = fixture()
  z.cfg.clutch_enabled = True
  z.moving_statistics = torch.zeros(n,8)
  z.metrics = defaultdict(lambda: torch.zeros(n))
  z.desired_wrist_pos_w = z.robot_wrist_pos_w.clone() + .01
  z.desired_wrist_quat_w = z.robot_wrist_quat_w.clone()
  z.foot_body_ids = [0,1]
  z.shoulder_height_difference = torch.zeros(n)
  z.torso_forward_axis_z = torch.zeros(n)
  z.robot.data.body_link_pos_w = z.robot_wrist_pos_w
  z.robot.data.root_link_lin_vel_b = torch.zeros(n,3)
  z.robot.data.root_link_ang_vel_b = torch.zeros(n,3)
  t = NS(mode=torch.full((n,),2), command=torch.tensor([.04,0.,.06]).repeat(n,1))
  z._env.command_manager = NS(get_term=lambda _:t)
  for _ in range(20):
    scope["_update_metrics"](z)
  t.command.zero_()
  for _ in range(100):
    scope["_update_metrics"](z)
  means = {"Metrics/wrists/"+key:float(value.mean()) for key,value in z.metrics.items()}
  result = conditional_metrics(means)
  assert abs(result["adjust_moving_velocity_xy_error"] - .04) < 1e-6
  assert abs(result["adjust_moving_velocity_yaw_error"] - .06) < 1e-6
  assert abs(result["adjust_moving_command_xy"] - .04) < 1e-6
  assert result["adjust_moving_wrist_error"] > .01
  print("PASS moving-only errors survive post-adjustment stationary snapshots")


if __name__ == "__main__":
  main()
