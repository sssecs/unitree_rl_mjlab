#!/usr/bin/env python3
"""CPU regression checks of deployed sampling/initialization, without simulation.

Extract the actual class methods via AST to avoid importing the task registry
and initializing simulator/GPU libraries. Identity root rotations are used in
these tests; this is not a dynamics or full-pose feasibility evaluation.
"""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from summarize_wrist_training import conditional_metrics


def identity_apply(quat, vector):
  assert torch.allclose(quat[..., 0], torch.ones_like(quat[..., 0]))
  assert torch.count_nonzero(quat[..., 1:]) == 0
  return vector


def identity_delta_mul(delta, quat):
  assert torch.allclose(delta[..., 0], torch.ones_like(delta[..., 0]))
  assert torch.count_nonzero(delta[..., 1:]) == 0
  return quat


def from_angle_axis(angle, axis):
  return torch.cat((torch.cos(angle / 2)[..., None], axis * torch.sin(angle / 2)[..., None]), -1)


def fixture(shoulder_range=(0.78, 0.98), step=90000):
  n = 1024
  cfg = NS(
    clutch_enabled=False,
    curriculum_warmup_steps=30000, curriculum_ramp_steps=60000,
    reach_probability=0.0, asymmetric_probability=0.0, height_probability=1.0,
    extension_range=(0.12, 0.28), shoulder_height_range=shoulder_range,
    wrist_height_offset_range=(-0.02, 0.02), low_reach_extension_range=(0.06, 0.14),
    lateral_offset_range=(-0.1, 0.1), vertical_offset_range=(-0.06, 0.06),
    orientation_angle_range=(-0.25, 0.25), reach_delay_s=1.0, reach_duration_s=2.0,
  )
  x = NS(scripted=False, cfg=cfg, _env=NS(common_step_counter=step, step_dt=4.0), device="cpu")
  for name in ("scenario", "extension", "height_difficulty", "sampled_shoulder_height",
               "sampled_wrist_height_offset", "elapsed", "phase", "start_shoulder_height",
               "target_shoulder_height", "final_shoulder_height"):
    setattr(x, name, torch.zeros(n))
  for name in ("height_active", "is_asymmetric", "needs_initialization"):
    setattr(x, name, torch.zeros(n, dtype=torch.bool))
  for name in ("sampled_offset_b", "sampled_axis_angle_b", "target_lin_vel_w", "start_pos_w",
               "target_pos_w", "previous_target_pos_w", "final_pos_w"):
    setattr(x, name, torch.zeros(n, 2, 3))
  identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(n, 2, 1)
  for name in ("start_quat_w", "target_quat_w", "final_quat_w", "robot_wrist_quat_w"):
    setattr(x, name, identity.clone())
  x.robot_wrist_pos_w = torch.zeros(n, 2, 3)
  x.robot_wrist_pos_w[..., 2] = 0.743
  x.shoulder_height = torch.full((n,), 1.091)
  x.robot = NS(data=NS(root_link_quat_w=identity[:, 0]))
  return x


def main():
  source = Path(__file__).resolve().parents[1] / "src/tasks/wrist_recovery/mdp/commands.py"
  tree = ast.parse(source.read_text())
  cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BimanualWristCommand")
  methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and
             node.name in ("_resample_command", "_update_command")]
  scope = {"torch": torch, "quat_apply": identity_apply,
           "quat_mul": identity_delta_mul, "quat_from_angle_axis": from_angle_axis}
  exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), scope)
  sample, update = scope["_resample_command"], scope["_update_command"]
  torch.manual_seed(1701)
  ids = torch.arange(0, 1024, 2).flip(0)  # Noncontiguous, permuted reset subset.
  for bounds in ((0.78, 0.98), (0.62, 0.90)):
    x = fixture(bounds)
    sample(x, ids)
    values = x.sampled_shoulder_height[ids]
    assert values.min() >= bounds[0] and values.max() <= bounds[1]
    assert values.std() > 0.03
    residual = x.sampled_wrist_height_offset[ids]
    assert residual.min() < 0 and residual.max() > 0
    assert torch.count_nonzero(x.sampled_shoulder_height[1::2]) == 0
    update(x)
    assert torch.allclose(x.final_shoulder_height[ids], values)
    expected = 0.743 + values - 1.091 + residual
    assert torch.allclose(x.final_pos_w[ids, :, 2], expected[:, None].expand(-1, 2))
    assert torch.allclose(x.target_shoulder_height[ids], values)
    assert torch.allclose(x.target_pos_w[ids], x.final_pos_w[ids])
    print(f"PASS range={bounds}; sampled shoulder={values.min():.4f}..{values.max():.4f}; "
          f"final wrist z={expected.min():.4f}..{expected.max():.4f}")
  for step, scale in ((30000, 0.0), (60000, 0.5), (90000, 1.0)):
    x = fixture(step=step)
    sample(x, ids)
    update(x)
    active = x.height_active.nonzero().flatten()
    if len(active):
      expected = 1.091 + scale * (x.sampled_shoulder_height[active] - 1.091)
      assert torch.allclose(x.final_shoulder_height[active], expected)
    else:
      assert scale == 0.0
  x = fixture()
  x.cfg.height_probability = 0.0
  x.cfg.reach_probability = 1.0
  sample(x, ids)
  assert x.sampled_offset_b[ids, :, 0].min() >= 0.12
  assert x.sampled_offset_b[ids, :, 0].max() <= 0.28
  sample(x, ids)  # Resampling must not retain stale extension values.
  assert x.sampled_offset_b[ids, :, 0].min() >= 0.12
  synthetic = {"Metrics/wrists/height_command_fraction": 0.25,
               "Metrics/wrists/height_wrist_pos_error_masked": 0.03,
               "Metrics/wrists/nonheight_wrist_pos_error_masked": 0.003}
  result = conditional_metrics(synthetic)
  assert result["height_wrist_pos_error"] == 0.12
  assert result["nonheight_wrist_pos_error"] == 0.004
  synthetic["Metrics/wrists/height_command_fraction"] = 0.0
  assert conditional_metrics(synthetic)["height_wrist_pos_error"] is None
  print("PASS curriculum, symmetric reach, reset subset isolation, and conditional normalization")


if __name__ == "__main__":
  main()
