"""Procedural manipulation commands, not joint motion references."""
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_from_angle_axis, quat_mul, quat_error_magnitude


def sample_clip(offset_b, root_quat, goal_quat, height_active, difficulty, cfg):
  n = offset_b.shape[0]
  delta = torch.zeros(n, 4, 2, 3, device=offset_b.device, dtype=offset_b.dtype)
  for k in range(1, 4):
    candidate = offset_b[..., :2] + (torch.rand(n, 2, 2, device=offset_b.device)*2-1)*.06*difficulty[:, None, None]
    for mask, xr, yr in ((height_active, cfg.low_reach_extension_range, cfg.height_lateral_offset_range),
                         (~height_active, cfg.extension_range, cfg.lateral_offset_range)):
      candidate[mask, :, 0] = torch.maximum(torch.minimum(candidate[mask, :, 0], xr[1]*difficulty[mask, None]), xr[0]*difficulty[mask, None])
      candidate[mask, :, 1] = torch.maximum(torch.minimum(candidate[mask, :, 1], yr[1]*difficulty[mask, None]), yr[0]*difficulty[mask, None])
    difference = candidate-offset_b[..., :2]
    scale = (.06*difficulty[:, None]/torch.linalg.vector_norm(difference, dim=-1).clamp_min(1e-6)).clamp(max=1)
    delta[:, k, :, :2] = difference*scale[..., None]
  q = root_quat[:, None, None, :].expand(-1, 4, 2, -1)
  delta = quat_apply(q, delta)
  delta[..., 2] = 0
  delta[:, 1, :, 2] = torch.rand(n, 2, device=delta.device)*.04*difficulty[:, None]
  delta[:, 2, :, 2] = (.08+torch.rand(n, 2, device=delta.device)*.04)*difficulty[:, None]
  axes = torch.randn(n, 4, 2, 3, device=delta.device, dtype=offset_b.dtype)
  axes /= torch.linalg.vector_norm(axes, dim=-1, keepdim=True).clamp_min(1e-6)
  angles = (torch.rand(n, 4, 2, device=delta.device)*2-1)*.35*difficulty[:, None, None]
  angles[:, 0] = 0
  quats = quat_mul(quat_from_angle_axis(angles, axes), goal_quat[:, None, :].expand(-1, 4, -1, -1))
  distance = torch.linalg.vector_norm(delta[:, 1:]-delta[:, :-1], dim=-1)
  angle = quat_error_magnitude(quats[:, 1:], quats[:, :-1])
  durations = 2+torch.rand(n, 3, 2, device=delta.device)
  for bound in (2*distance/.12, torch.sqrt(6*distance/.30), 2*angle/.60, torch.sqrt(9*angle/1.50)):
    durations = torch.maximum(durations, bound)
  return delta, quats, durations


def continue_clip(delta, quats, durations, done, offset_b, root_quat, goal_quat,
                  height_active, difficulty, cfg):
  """Replace only completed hands, retaining endpoint pose and zero velocity."""
  new_delta, new_quats, new_duration = sample_clip(offset_b, root_quat, goal_quat, height_active, difficulty, cfg)
  new_delta[:, 0] = delta[:, -1]
  new_quats[:, 0] = quats[:, -1]
  pause = torch.rand_like(difficulty[:, None].expand(-1, 2)) < .20
  new_delta[:, 1] = torch.where(pause[..., None], new_delta[:, 0], new_delta[:, 1])
  new_quats[:, 1] = torch.where(pause[..., None], new_quats[:, 0], new_quats[:, 1])
  distance = torch.linalg.vector_norm(new_delta[:, 1:]-new_delta[:, :-1], dim=-1)
  angle = quat_error_magnitude(new_quats[:, 1:], new_quats[:, :-1])
  for bound in (2*distance/.12, torch.sqrt(6*distance/.30), 2*angle/.60, torch.sqrt(9*angle/1.50)):
    new_duration = torch.maximum(new_duration, bound)
  delta.copy_(torch.where(done[:, None, :, None], new_delta, delta))
  quats.copy_(torch.where(done[:, None, :, None], new_quats, quats))
  durations.copy_(torch.where(done[:, None, :], new_duration, durations))


def evaluate_clip(delta, quats, durations, age):
  # Independent hand timing; clamp at the last pose rather than wrap/repeat.
  elapsed = age[:, None] if age.ndim == 1 else age
  ends = durations.cumsum(1)
  segment = (elapsed[:, None, :] >= ends[:, :2]).sum(1).clamp(max=2)
  rows = torch.arange(len(age), device=age.device)[:, None]
  hands = torch.arange(2, device=age.device)[None, :]
  starts = torch.cat((torch.zeros_like(ends[:, :1]), ends[:, :-1]), dim=1)
  u = ((elapsed-starts[rows, segment, hands])/durations[rows, segment, hands]).clamp(0, 1)
  blend = u.pow(3)*(10-15*u+6*u.square())
  left, right = delta[rows, segment, hands], delta[rows, segment+1, hands]
  position = left+(right-left)*blend[..., None]
  q0, q1 = quats[rows, segment, hands], quats[rows, segment+1, hands]
  q1 = torch.where((q0*q1).sum(-1, keepdim=True)<0, -q1, q1)
  orientation = q0+(q1-q0)*blend[..., None]
  orientation /= torch.linalg.vector_norm(orientation, dim=-1, keepdim=True).clamp_min(1e-6)
  return position, orientation, segment
