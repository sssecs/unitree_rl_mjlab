#!/usr/bin/env python3
"""Plot one EgoDex clip and its fixed shoulder-ground-frame wrist mapping.

The source coordinate frame is created once from the selected clip's first
frame: origin = the shoulder-midpoint projected to the ground, up = ARKit +Y,
right = left-to-right shoulder direction projected to the ground.  It is not
updated as the person bends or moves.  The target frame is G1 world coordinates
with +X forward, +Y left, +Z up, and its origin at ground level.

Examples:
  python tools/plot_egodex_mapping.py --trajectory basic_fold/53.hdf5 --show
  python tools/plot_egodex_mapping.py --trajectory tie_and_untie_shoelace/14.hdf5 \\
      --start 20 --stop 250 --output /tmp/egodex_mapping.png
  python tools/plot_egodex_mapping.py --trajectory basic_fold/53.hdf5 \\
      --stride 2 --animation /tmp/egodex_mapping.gif --fps 15
  python tools/plot_egodex_mapping.py --trajectory basic_fold/53.hdf5 \\
      --animate --show --fps 30
"""
from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import matplotlib.pyplot as plt
from matplotlib import animation as mpl_animation
import numpy as np


DEFAULT_ROOT = Path("/mnt/hdd/humanoid_locomotion/datasets/data/egodex_hdf5/test")
UP_H = np.array([0.0, 1.0, 0.0])
FORWARD_G = np.array([1.0, 0.0, 0.0])
RIGHT_G = np.array([0.0, -1.0, 0.0])
UP_G = np.array([0.0, 0.0, 1.0])
G1_XML = Path(__file__).resolve().parents[1] / "src/assets/robots/unitree_g1/xmls/g1.xml"

# A compact torso-and-arm skeleton sufficient to inspect the mapping.  Missing
# joints are skipped because a few clips contain only the required joints.
EDGES = (
    ("hip", "spine1"), ("spine1", "spine2"), ("spine2", "spine3"),
    ("spine3", "spine4"), ("spine4", "spine5"), ("spine5", "spine6"),
    ("spine6", "spine7"), ("spine7", "neck1"), ("neck1", "neck2"),
    ("neck2", "neck3"), ("neck3", "neck4"),
    ("neck4", "leftShoulder"), ("leftShoulder", "leftArm"),
    ("leftArm", "leftForearm"), ("leftForearm", "leftHand"),
    ("neck4", "rightShoulder"), ("rightShoulder", "rightArm"),
    ("rightArm", "rightForearm"), ("rightForearm", "rightHand"),
)


def position(data: h5py.File, name: str) -> np.ndarray:
  return np.asarray(data[f"transforms/{name}"][:, :3, 3], dtype=np.float64)


def normalize(vector: np.ndarray, label: str) -> np.ndarray:
  length = float(np.linalg.norm(vector))
  if not np.isfinite(length) or length < 1.0e-6:
    raise ValueError(f"Cannot construct mapping frame: {label} is degenerate")
  return vector / length


def source_basis(left_shoulder: np.ndarray, right_shoulder: np.ndarray) -> np.ndarray:
  """Return columns [forward, right, up] in the fixed human world frame."""
  right = right_shoulder - left_shoulder
  right -= right.dot(UP_H) * UP_H
  right = normalize(right, "horizontal shoulder line")
  forward = normalize(np.cross(UP_H, right), "forward from shoulder line")
  return np.column_stack((forward, right, UP_H))


def mapped_positions(points_h: np.ndarray, origin_h: np.ndarray, basis_h: np.ndarray,
                     scale: float) -> np.ndarray:
  """Map arbitrary source world points into the fixed G1 ground frame."""
  basis_g = np.column_stack((FORWARD_G, RIGHT_G, UP_G))
  rotation = basis_g @ basis_h.T
  return scale * np.einsum("ij,...j->...i", rotation, points_h - origin_h)


def human_arm_length(joints: dict[str, np.ndarray]) -> float:
  """Median shoulder-to-hand chain length in the selected EgoDex clip."""
  lengths = []
  for side in ("left", "right"):
    names = (f"{side}Shoulder", f"{side}Arm", f"{side}Forearm", f"{side}Hand")
    if any(name not in joints for name in names):
      raise ValueError(f"Missing {side} arm joints needed for scale calculation")
    chain = sum(np.linalg.norm(joints[parent] - joints[child], axis=1)
                for parent, child in zip(names, names[1:]))
    lengths.append(chain)
  return float(np.median(np.concatenate(lengths)))


def direct_child(node: ET.Element, name: str) -> ET.Element:
  for child in node.findall("body"):
    if child.get("name") == name:
      return child
  raise ValueError(f"G1 XML arm chain is missing body {name}")


def g1_arm_length(xml_path: Path = G1_XML) -> float:
  """Kinematic shoulder-pitch to wrist-yaw chain length from the shipped XML."""
  worldbody = ET.parse(xml_path).getroot().find("worldbody")
  if worldbody is None:
    raise ValueError(f"G1 XML has no worldbody: {xml_path}")
  start_name = "left_shoulder_pitch_link"
  start = next((node for node in worldbody.iter("body") if node.get("name") == start_name), None)
  if start is None:
    raise ValueError(f"G1 XML is missing body {start_name}")
  chain = ("left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link",
           "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link")
  length, node = 0.0, start
  for name in chain:
    node = direct_child(node, name)
    offset = np.fromstring(node.get("pos", "0 0 0"), sep=" ")
    if offset.shape != (3,):
      raise ValueError(f"Invalid G1 XML offset for {name}: {node.get('pos')}")
    length += float(np.linalg.norm(offset))
  return length


def equal_axes(axis, points: np.ndarray) -> None:
  lower, upper = points.min(0), points.max(0)
  radius = max(float((upper - lower).max()) * 0.55, 0.15)
  center = (lower + upper) / 2
  axis.set(xlim=(center[0] - radius, center[0] + radius),
           ylim=(center[1] - radius, center[1] + radius),
           zlim=(center[2] - radius, center[2] + radius))


def plot_skeleton(axis, joints: dict[str, np.ndarray], frame: int, color: str,
                  alpha: float, label: str | None = None) -> None:
  for parent, child in EDGES:
    if parent in joints and child in joints:
      segment = np.stack((joints[parent][frame], joints[child][frame]))
      axis.plot(*segment.T, color=color, alpha=alpha, linewidth=1.4)
  visible = np.stack([values[frame] for values in joints.values()])
  axis.scatter(*visible.T, color=color, alpha=alpha, s=9, label=label)


def resolve_trajectory(value: str, root: Path) -> Path:
  path = Path(value)
  if not path.is_absolute():
    path = root / path
  if path.suffix not in (".hdf5", ".h5"):
    raise ValueError("--trajectory must name an .hdf5 or .h5 file")
  if not path.is_file():
    raise FileNotFoundError(path)
  return path


def configure_human_axis(axis, title: str, points: np.ndarray) -> None:
  axis.set(title=title, xlabel="ARKit X (m)", ylabel="ARKit Y / up (m)", zlabel="ARKit Z (m)")
  equal_axes(axis, points)


def configure_mapped_axis(axis, title: str, points: np.ndarray) -> None:
  axis.set(title=title, xlabel="G1 forward X (m)", ylabel="G1 left Y (m)", zlabel="G1 up Z (m)")
  equal_axes(axis, points)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--trajectory", required=True,
                      help="absolute file or path relative to --dataset-root")
  parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
  parser.add_argument("--start", type=int, default=0, help="inclusive frame index")
  parser.add_argument("--stop", type=int, default=None, help="exclusive frame index")
  parser.add_argument("--stride", type=int, default=1, help="plot every Nth frame")
  parser.add_argument("--output", type=Path, default=None, help="write PNG here")
  parser.add_argument("--animation", type=Path, default=None,
                      help="write a .gif or .mp4 3D animation here")
  parser.add_argument("--animate", action="store_true",
                      help="play the 3D animation interactively (requires --show)")
  parser.add_argument("--fps", type=int, default=15, help="animation frame rate (default: 15)")
  parser.add_argument("--show", action="store_true", help="open an interactive window")
  args = parser.parse_args()
  if args.start < 0 or args.stride < 1:
    parser.error("--start >= 0 and --stride >= 1 are required")
  if args.fps < 1:
    parser.error("--fps must be positive")
  if args.animate and not args.show:
    parser.error("--animate requires --show")
  if args.output is None and args.animation is None and not args.show:
    parser.error("Choose --show, --output, --animation, or a combination")
  if args.animation and args.animation.suffix.lower() not in (".gif", ".mp4"):
    parser.error("--animation must end in .gif or .mp4")

  path = resolve_trajectory(args.trajectory, args.dataset_root)
  required = ("hip", "leftShoulder", "rightShoulder", "leftHand", "rightHand")
  with h5py.File(path, "r") as data:
    missing = [name for name in required if f"transforms/{name}" not in data]
    if missing:
      raise ValueError(f"Missing required transforms: {missing}")
    joints = {name: position(data, name) for name in data["transforms"].keys()}
    count = len(joints["hip"])
    if any(len(values) != count or not np.isfinite(values).all() for values in joints.values()):
      raise ValueError("Transforms have inconsistent lengths or non-finite positions")

  stop = count if args.stop is None else min(args.stop, count)
  if not 0 <= args.start < stop:
    raise ValueError(f"Selected frame range [{args.start}, {stop}) is empty for {count} frames")
  indices = np.arange(args.start, stop, args.stride)
  # The mapping frame deliberately comes from the selected trajectory's first
  # frame, not from args.start: inspecting a subrange must not redefine targets.
  shoulder_mid = (joints["leftShoulder"] + joints["rightShoulder"]) / 2
  origin_h = shoulder_mid[0].copy()
  origin_h[1] = 0.0
  basis_h = source_basis(joints["leftShoulder"][0], joints["rightShoulder"][0])
  source_arm_length = human_arm_length(joints)
  target_arm_length = g1_arm_length()
  scale = target_arm_length / source_arm_length
  wrists_h = np.stack((joints["leftHand"], joints["rightHand"]), axis=1)
  mapped_joints = {
      name: mapped_positions(values, origin_h, basis_h, scale)
      for name, values in joints.items()
  }
  wrists_g = mapped_positions(wrists_h, origin_h, basis_h, scale)
  shoulder_g = mapped_positions(shoulder_mid, origin_h, basis_h, scale)
  human_points = np.concatenate((np.stack([values[indices] for values in joints.values()]).reshape(-1, 3), origin_h[None]))
  mapped_points = np.concatenate((
      np.stack([values[indices] for values in mapped_joints.values()]).reshape(-1, 3),
      wrists_g[indices].reshape(-1, 3), shoulder_g[indices],
  ))
  mapped_title = (f"Fixed-frame G1 targets (arm scale={scale:.4f}; "
                  f"{target_arm_length:.3f}/{source_arm_length:.3f} m)")
  print(f"EgoDex shoulder-to-hand chain: {source_arm_length:.6f} m")
  print(f"G1 shoulder-pitch-to-wrist-yaw chain: {target_arm_length:.6f} m")
  print(f"Applied arm-length scale: {scale:.6f}")

  if args.animation or args.animate:
    figure = plt.figure(figsize=(9, 8), constrained_layout=True)
    mapped_axis = figure.add_subplot(1, 1, 1, projection="3d")

    def update(index_position: int):
      frame = int(indices[index_position])
      mapped_axis.cla()
      plot_skeleton(mapped_axis, mapped_joints, frame, "#7f7f7f", 0.72,
                    "current mapped EgoDex skeleton")
      mapped_axis.scatter(0, 0, 0, color="black", marker="x", s=55,
                          label="shoulder-ground origin")
      for hand, (label, color) in enumerate(zip(("left wrist target", "right wrist target"),
                                                  ("#1f77b4", "#d62728"))):
        trace = wrists_g[indices[:index_position + 1], hand]
        mapped_axis.plot(*trace.T, color=color, linewidth=1.8, label=label)
        mapped_axis.scatter(*trace[-1], color=color, marker="o", s=35)
      shoulder_trace = shoulder_g[indices[:index_position + 1]]
      mapped_axis.plot(*shoulder_trace.T, color="#2ca02c", linestyle="--", linewidth=1.4,
                       label="shoulder-midpoint target")
      configure_mapped_axis(mapped_axis, f"{mapped_title}; frame {frame}/{count - 1}", mapped_points)
      mapped_axis.legend(loc="upper left", fontsize=8)
      return ()

    movie = mpl_animation.FuncAnimation(figure, update, frames=len(indices), interval=1000 / args.fps,
                                        blit=False, repeat=True)
    if args.animation:
      args.animation.parent.mkdir(parents=True, exist_ok=True)
      if args.animation.suffix.lower() == ".gif":
        movie.save(args.animation, writer=mpl_animation.PillowWriter(fps=args.fps), dpi=110)
      else:
        if not mpl_animation.writers.is_available("ffmpeg"):
          raise RuntimeError("MP4 needs ffmpeg; use a .gif output or install ffmpeg outside this project")
        movie.save(args.animation, writer=mpl_animation.FFMpegWriter(fps=args.fps), dpi=110)
      print(args.animation.resolve())
    if args.show:
      plt.show()
    return

  figure = plt.figure(figsize=(15, 7), constrained_layout=True)
  human_axis = figure.add_subplot(1, 2, 1, projection="3d")
  mapped_axis = figure.add_subplot(1, 2, 2, projection="3d")
  key_frames = sorted({int(indices[0]), int(indices[len(indices) // 2]), int(indices[-1])})
  for frame, color, alpha, label in zip(key_frames, ("#1f77b4", "#ff7f0e", "#d62728"),
                                         (0.32, 0.55, 0.9), ("start", "middle", "end")):
    plot_skeleton(human_axis, joints, frame, color, alpha, label)
  human_axis.scatter(*origin_h, color="black", marker="x", s=55, label="fixed ground origin")
  configure_human_axis(human_axis, f"EgoDex skeleton: {path.relative_to(args.dataset_root)}", human_points)
  human_axis.legend(loc="upper left", fontsize=8)

  labels = ("left wrist target", "right wrist target")
  colors = ("#1f77b4", "#d62728")
  for hand, (label, color) in enumerate(zip(labels, colors)):
    trace = wrists_g[indices, hand]
    mapped_axis.plot(*trace.T, color=color, linewidth=1.8, label=label)
    mapped_axis.scatter(*trace[0], color=color, marker="o", s=32)
    mapped_axis.scatter(*trace[-1], color=color, marker="x", s=42)
  mapped_axis.plot(*shoulder_g[indices].T, color="#2ca02c", linestyle="--", linewidth=1.4,
                   label="shoulder-midpoint target")
  mapped_axis.plot((0, 0), (0, 0), (0, float(np.max(shoulder_g[indices, 2]))),
                   color="black", alpha=0.25, linewidth=1)
  configure_mapped_axis(mapped_axis, mapped_title, mapped_points)
  mapped_axis.legend(loc="upper left", fontsize=8)

  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print(args.output.resolve())
  if args.show:
    plt.show()


if __name__ == "__main__":
  main()
