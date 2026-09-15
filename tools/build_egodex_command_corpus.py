#!/usr/bin/env python3
"""Extract compact fixed-frame EgoDex wrist/shoulder commands for RL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/mnt/hdd/humanoid_locomotion/datasets/data/egodex_hdf5/test")
UP = np.array([0., 1., 0.])
G1_ARM = .41039406453435195


def unit(v: np.ndarray) -> np.ndarray:
  n = np.linalg.norm(v)
  if not np.isfinite(n) or n < 1e-6:
    raise ValueError("degenerate shoulder line")
  return v / n


def quat_wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
  """Convert proper rotation matrices with shape (..., 3, 3) to wxyz quaternions."""
  matrix = np.asarray(matrix, dtype=np.float64)
  flat = matrix.reshape(-1, 3, 3)
  out = np.empty((len(flat), 4), dtype=np.float64)
  trace = np.trace(flat, axis1=1, axis2=2)
  positive = trace > 0
  scale = np.sqrt(np.maximum(trace[positive] + 1.0, 1e-12)) * 2
  out[positive, 0] = .25 * scale
  out[positive, 1] = (flat[positive, 2, 1] - flat[positive, 1, 2]) / scale
  out[positive, 2] = (flat[positive, 0, 2] - flat[positive, 2, 0]) / scale
  out[positive, 3] = (flat[positive, 1, 0] - flat[positive, 0, 1]) / scale
  for diagonal in range(3):
    mask = ~positive & (np.argmax(np.diagonal(flat, axis1=1, axis2=2), axis=1) == diagonal)
    if not np.any(mask):
      continue
    i, j, k = diagonal, (diagonal + 1) % 3, (diagonal + 2) % 3
    scale = np.sqrt(np.maximum(1.0 + flat[mask, i, i] - flat[mask, j, j] - flat[mask, k, k], 1e-12)) * 2
    out[mask, 0] = (flat[mask, k, j] - flat[mask, j, k]) / scale
    out[mask, i + 1] = .25 * scale
    out[mask, j + 1] = (flat[mask, j, i] + flat[mask, i, j]) / scale
    out[mask, k + 1] = (flat[mask, k, i] + flat[mask, i, k]) / scale
  out /= np.linalg.norm(out, axis=1, keepdims=True).clip(1e-12)
  return out.reshape(*matrix.shape[:-2], 4).astype(np.float32)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--root", type=Path, default=ROOT)
  ap.add_argument("--output", type=Path, required=True)
  ap.add_argument("--source-fps", type=float, default=30.)
  ap.add_argument("--max-files", type=int, default=None)
  args = ap.parse_args()
  if args.source_fps <= 0: ap.error("--source-fps must be positive")
  files = sorted(args.root.rglob("*.hdf5"))
  if args.max_files is not None: files = files[:args.max_files]
  chunks, orientation_chunks, offsets, lengths, rejected, speeds = [], [], [0], [], [], []
  required = ("leftShoulder", "leftArm", "leftForearm", "leftHand",
              "rightShoulder", "rightArm", "rightForearm", "rightHand")
  for index, path in enumerate(files, 1):
    try:
      with h5py.File(path, "r") as f:
        transforms = {name: np.asarray(f[f"transforms/{name}"], dtype=np.float32) for name in required}
      p = {name: value[:, :3, 3] for name, value in transforms.items()}
      if not all(len(v) >= 2 and np.isfinite(v).all() for v in p.values()): raise ValueError("invalid positions")
      shoulder = (p["leftShoulder"] + p["rightShoulder"]) / 2
      origin = shoulder[0].copy(); origin[1] = 0
      right = p["rightShoulder"][0] - p["leftShoulder"][0]; right -= right.dot(UP) * UP
      right = unit(right); forward = unit(np.cross(UP, right)); basis = np.column_stack((forward, right, UP))
      arm_lengths = []
      for side in ("left", "right"):
        names = (f"{side}Shoulder", f"{side}Arm", f"{side}Forearm", f"{side}Hand")
        arm_lengths.append(sum(np.linalg.norm(p[a] - p[b], axis=1) for a,b in zip(names, names[1:])))
      source_arm = float(np.median(np.concatenate(arm_lengths)))
      if source_arm < .1: raise ValueError("invalid arm length")
      scale = G1_ARM / source_arm
      world = np.stack((p["leftHand"], p["rightHand"], shoulder), axis=1)
      local = np.einsum("ji,tkj->tki", basis, world - origin).astype(np.float32) * scale
      hand_rotations = np.stack((transforms["leftHand"][:, :3, :3], transforms["rightHand"][:, :3, :3]), axis=1)
      if not np.allclose(hand_rotations.transpose(0, 1, 3, 2) @ hand_rotations, np.eye(3), atol=2e-3):
        raise ValueError("invalid hand rotation matrix")
      # The source frame is fixed at clip start, exactly as for positions.
      local_quat = quat_wxyz_from_matrix(np.einsum("ji,thjk->thik", basis, hand_rotations))
      chunks.append(local); lengths.append(len(local)); offsets.append(offsets[-1] + len(local))
      orientation_chunks.append(local_quat)
      speeds.append(np.linalg.norm(np.diff(local[:, :2], axis=0), axis=-1).reshape(-1) * args.source_fps)
    except (OSError, KeyError, ValueError) as exc:
      rejected.append(f"{path.relative_to(args.root)}: {exc}")
    if index % 250 == 0 or index == len(files): print(f"{index}/{len(files)} valid={len(chunks)} rejected={len(rejected)}", flush=True)
  if not chunks: raise RuntimeError("no usable trajectories")
  positions = np.concatenate(chunks)
  orientations = np.concatenate(orientation_chunks)
  speed = np.concatenate(speeds)
  meta = {"source_root": str(args.root), "source_fps": args.source_fps, "trajectories": len(chunks),
          "frames": int(len(positions)), "source_arm_m": float(source_arm), "g1_arm_m": G1_ARM,
          "scale_definition": "per-clip G1 arm-chain / shoulder-to-hand median distance",
          "orientation_definition": "absolute ARKit hand rotation in the fixed shoulder-ground frame; runtime maps it to the matching fixed G1 shoulder-ground frame without per-clip or per-reset orientation alignment",
          "speed_mps": {k: float(np.quantile(speed, q)) for k,q in (("p50",.5),("p95",.95),("p99",.99),("max",1.))},
          "rejected": rejected}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(args.output, positions=positions, orientations=orientations, offsets=np.asarray(offsets, np.int64),
                      lengths=np.asarray(lengths, np.int32), metadata=json.dumps(meta))
  print(json.dumps(meta, indent=2)); print(args.output)


if __name__ == "__main__": main()
