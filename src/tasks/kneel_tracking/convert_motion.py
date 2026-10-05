"""Convert optimizer or GMR G1 NPZ poses to 50 Hz FK references."""

import argparse
import hashlib
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

DATA_DIR = Path(__file__).parent / "data"


def convert(source: Path, destination: Path, fps: int = 50, hold_s: float = 1.0):
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  with np.load(source, allow_pickle=False) as source_data:
    has_contacts = "contact_mode" in source_data
    if "qpos" in source_data:
      times = source_data["times"].copy()
      knots = source_data["qpos"].copy()
    else:
      # GMR NPZ export uses xyzw root quaternions and named hinge joints.
      source_fps = float(source_data["fps"])
      if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("GMR fps must be finite and positive")
      joint_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
      names = [model.joint(int(i)).name for i in joint_ids]
      source_names = source_data["joint_names"].tolist()
      if len(set(source_names)) != len(names) or set(source_names) != set(names):
        raise ValueError("GMR joint names must match the 29-DoF G1 model")
      joints = source_data["dof_pos"][:, [source_names.index(name) for name in names]]
      knots = np.concatenate([
        source_data["root_pos"], source_data["root_rot"][:, [3, 0, 1, 2]], joints,
      ], axis=1)
      times = np.arange(len(knots)) / source_fps
    modes = (source_data["contact_mode"].copy() if has_contacts
             else np.full(len(times), "unknown"))
  if len(times) < 2 or knots.shape != (len(times), model.nq):
    raise ValueError("Reference qpos must match the 29-DoF G1 model")
  if not np.isfinite(knots).all() or not np.all(np.diff(times) > 0):
    raise ValueError("Expected finite poses and strictly increasing times")
  if fps <= 0 or hold_s < 0 or times[0] != 0:
    raise ValueError("Expected positive fps, nonnegative hold, and times starting at zero")
  if modes.shape != times.shape:
    raise ValueError("Expected one contact mode per source frame")
  sample_times = np.arange(round((times[-1] + hold_s) * fps) + 1) / fps
  clipped_times = np.minimum(sample_times, times[-1])
  qpos = np.stack(
    [np.interp(clipped_times, times, knots[:, i]) for i in range(model.nq)], axis=1
  )
  rotation = Slerp(times, Rotation.from_quat(knots[:, [4, 5, 6, 3]]))
  qpos[:, 3:7] = rotation(clipped_times).as_quat()[:, [3, 0, 1, 2]]
  dt = 1.0 / fps
  qvel = np.zeros((len(qpos), model.nv))
  for i in range(len(qpos)):
    left, right = max(0, i - 1), min(len(qpos) - 1, i + 1)
    mujoco.mj_differentiatePos(model, qvel[i], (right - left) * dt, qpos[left], qpos[right])

  data = mujoco.MjData(model)
  positions, quaternions = [], []
  for pose in qpos:
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    # Entity body indices exclude MuJoCo's world body.
    positions.append(data.xpos[1:].copy())
    quaternions.append(data.xquat[1:].copy())
  positions = np.asarray(positions)
  quaternions = np.asarray(quaternions)
  linear_velocity = np.gradient(positions, dt, axis=0)
  angular_velocity = np.zeros_like(positions)
  for body in range(model.nbody - 1):
    rotations = Rotation.from_quat(quaternions[:, body, [1, 2, 3, 0]])
    for i in range(len(qpos)):
      left, right = max(0, i - 1), min(len(qpos) - 1, i + 1)
      angular_velocity[i, body] = (
        rotations[right] * rotations[left].inv()
      ).as_rotvec() / ((right - left) * dt)
  mode_indices = np.searchsorted(times, clipped_times, side="right") - 1
  sample_modes = modes[mode_indices]
  contact_targets = np.zeros((len(qpos), 4), dtype=np.float32)
  if has_contacts:
    contact_targets[:, :2] = 1  # Feet/forefeet remain support through kneel clip.
  contact_targets[:, 2] = np.isin(sample_modes, ["feet_left_knee", "feet_both_knees"])
  contact_targets[:, 3] = sample_modes == "feet_both_knees"
  joint_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
  joint_names = np.asarray([model.joint(int(i)).name for i in joint_ids])
  body_names = np.asarray([model.body(i).name for i in range(1, model.nbody)])
  destination.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    destination,
    fps=fps,
    times=sample_times,
    qpos=qpos,
    joint_pos=qpos[:, model.jnt_qposadr[joint_ids]].astype(np.float32),
    joint_vel=qvel[:, model.jnt_dofadr[joint_ids]].astype(np.float32),
    body_pos_w=positions.astype(np.float32),
    body_quat_w=quaternions.astype(np.float32),
    body_lin_vel_w=linear_velocity.astype(np.float32),
    body_ang_vel_w=angular_velocity.astype(np.float32),
    joint_names=joint_names,
    body_names=body_names,
    contact_mode=sample_modes,
    contact_targets=contact_targets,
    has_contact_labels=has_contacts,
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    model_sha256=hashlib.sha256(G1_XML.read_bytes()).hexdigest(),
    source_duration_s=times[-1],
    hold_s=hold_s,
  )
  print(f"Saved {destination}: {len(qpos)} frames, {fps} Hz, {sample_times[-1]:.2f} s")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", type=Path, default=DATA_DIR / "transition_original.npz")
  parser.add_argument("--output", type=Path, default=DATA_DIR / "kneel_50hz.npz")
  args = parser.parse_args()
  convert(args.source, args.output)
