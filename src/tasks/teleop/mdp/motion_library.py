from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
import torch


EXPECTED_INTERFACE_VERSION = "3.2"
STYLE_DESCRIPTOR_NAMES = (
  "pelvis_height_neutral_norm",
  "effective_leg_length_left",
  "effective_leg_length_right",
  "knee_ground_distance_left",
  "knee_ground_distance_right",
  "torso_pitch",
  "shoulder_mid_rel_pelvis_forward",
  "shoulder_mid_rel_pelvis_left",
  "shoulder_pelvis_yaw_difference",
  "left_foot_rel_pelvis_forward",
  "left_foot_rel_pelvis_left",
  "right_foot_rel_pelvis_forward",
  "right_foot_rel_pelvis_left",
  "foot_pseudo_contact_left",
  "foot_pseudo_contact_right",
  "knee_pseudo_contact_left",
  "knee_pseudo_contact_right",
)


def _np_quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
  """Hamilton product for scalar-first numpy quaternions [w, x, y, z]."""
  w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
  w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
  return np.stack(
    (
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ),
    axis=-1,
  )


def _quat_nlerp(
  q0: torch.Tensor,
  q1: torch.Tensor,
  alpha: torch.Tensor,
) -> torch.Tensor:
  """Shortest-path normalized linear quaternion interpolation."""
  dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
  q1 = torch.where(dot < 0.0, -q1, q1)

  a = alpha
  while a.ndim < q0.ndim:
    a = a.unsqueeze(-1)

  q = q0 + a * (q1 - q0)
  return torch.nn.functional.normalize(q, dim=-1)


class SparseWholeBodyMotionLibrary:
  """Recursive sparse-command NPZ library.

  Design follows TWIST's MotionLib pattern:
    * load many motions into one library,
    * keep a motion id per environment,
    * sample motion ids with torch.multinomial(replacement=True),
    * query the selected motion at an independent time for every environment.

  Unlike TWIST's fixed-FPS pickle format, our v3.2 NPZ clips retain their
  original timestamps. Frames are stored ragged/flat and a vectorized binary
  search finds interpolation indices for every environment.
  """

  REQUIRED_KEYS = (
    "timestamp",
    "left_wrist_pos_world",
    "left_wrist_quat_world_wxyz",
    "right_wrist_pos_world",
    "right_wrist_quat_world_wxyz",
    "torso_center_xy_world",
    "left_shoulder_height",
    "right_shoulder_height",
    "torso_heading_world",
  )

  def __init__(
    self,
    command_source: str,
    device: str,
    canonicalize_heading: bool = False,
    recursive: bool = True,
    skip_invalid_files: bool = True,
    sampling_weight_mode: Literal["uniform", "duration"] = "uniform",
    load_style_descriptors: bool = False,
  ) -> None:
    self.device = device
    self.source_path = Path(command_source).expanduser().resolve()
    self.recursive = recursive
    self.skip_invalid_files = skip_invalid_files
    self.sampling_weight_mode = sampling_weight_mode

    discovered = self._discover_npz_files(self.source_path, recursive)
    if not discovered:
      raise RuntimeError(
        f"No .npz files found under command source: {self.source_path}"
      )

    arrays: dict[str, list[np.ndarray]] = {
      "timestamp": [],
      "left_pos": [],
      "left_quat": [],
      "right_pos": [],
      "right_quat": [],
      "shoulder_mid_xy": [],
      "left_shoulder_h": [],
      "right_shoulder_h": [],
      "shoulder_heading": [],
    }
    if load_style_descriptors:
      arrays["style_descriptor"] = []
      arrays["style_valid"] = []
    valid_files: list[Path] = []
    num_frames: list[int] = []
    durations: list[float] = []
    invalid: list[tuple[Path, str]] = []
    version_counts: dict[str, int] = {}

    for path in discovered:
      try:
        clip, version = self._load_one_npz(
          path,
          canonicalize_heading=canonicalize_heading,
          load_style_descriptors=load_style_descriptors,
        )
      except Exception as exc:
        if not skip_invalid_files:
          raise RuntimeError(f"Invalid command NPZ {path}: {exc}") from exc
        invalid.append((path, str(exc)))
        continue

      for key, value in clip.items():
        arrays[key].append(value)
      valid_files.append(path)
      num_frames.append(int(clip["timestamp"].shape[0]))
      durations.append(float(clip["timestamp"][-1]))
      version_key = version or "unknown"
      version_counts[version_key] = version_counts.get(version_key, 0) + 1

    if not valid_files:
      detail = "\n".join(
        f"  {path}: {err}" for path, err in invalid[:10]
      )
      raise RuntimeError(
        "No valid sparse-command NPZ files were found."
        + (f"\nFirst errors:\n{detail}" if detail else "")
      )

    self.motion_files = valid_files
    if self.source_path.is_dir():
      self.motion_names = [
        str(path.relative_to(self.source_path)) for path in valid_files
      ]
    else:
      self.motion_names = [valid_files[0].name]

    frame_counts_np = np.asarray(num_frames, dtype=np.int64)
    start_idx_np = np.zeros(len(valid_files), dtype=np.int64)
    if len(valid_files) > 1:
      start_idx_np[1:] = np.cumsum(frame_counts_np[:-1])

    self.motion_num_frames = torch.tensor(
      frame_counts_np, dtype=torch.long, device=device
    )
    self.motion_start_idx = torch.tensor(
      start_idx_np, dtype=torch.long, device=device
    )
    self.motion_lengths = torch.tensor(
      durations, dtype=torch.float32, device=device
    )

    # All timestamps stay LOCAL to their own clip. The per-motion start index
    # tells the vectorized binary search which slice of this flat array to use.
    self.timestamp_flat = torch.tensor(
      np.concatenate(arrays["timestamp"], axis=0),
      dtype=torch.float32,
      device=device,
    )
    self.npz_left_wrist_pos_cw = torch.tensor(
      np.concatenate(arrays["left_pos"], axis=0),
      dtype=torch.float32,
      device=device,
    )
    self.npz_left_wrist_quat_cw = torch.tensor(
      np.concatenate(arrays["left_quat"], axis=0),
      dtype=torch.float32,
      device=device,
    )
    self.npz_right_wrist_pos_cw = torch.tensor(
      np.concatenate(arrays["right_pos"], axis=0),
      dtype=torch.float32,
      device=device,
    )
    self.npz_right_wrist_quat_cw = torch.tensor(
      np.concatenate(arrays["right_quat"], axis=0),
      dtype=torch.float32,
      device=device,
    )
    self.npz_shoulder_mid_xy_cw = torch.tensor(
      np.concatenate(arrays["shoulder_mid_xy"], axis=0),
      dtype=torch.float32,
      device=device,
    )
    self.npz_left_shoulder_height = torch.tensor(
      np.concatenate(arrays["left_shoulder_h"], axis=0),
      dtype=torch.float32,
      device=device,
    )
    self.npz_right_shoulder_height = torch.tensor(
      np.concatenate(arrays["right_shoulder_h"], axis=0),
      dtype=torch.float32,
      device=device,
    )
    self.npz_shoulder_heading_cw = torch.tensor(
      np.concatenate(arrays["shoulder_heading"], axis=0),
      dtype=torch.float32,
      device=device,
    )
    self.style_descriptor_flat: torch.Tensor | None = None
    self.style_valid_flat: torch.Tensor | None = None
    if load_style_descriptors:
      self.style_descriptor_flat = torch.tensor(
        np.concatenate(arrays["style_descriptor"], axis=0),
        dtype=torch.float32,
        device=device,
      )
      self.style_valid_flat = torch.tensor(
        np.concatenate(arrays["style_valid"], axis=0),
        dtype=torch.bool,
        device=device,
      )

    if sampling_weight_mode == "uniform":
      raw_weights = torch.ones(
        self.num_motions, dtype=torch.float32, device=device
      )
    elif sampling_weight_mode == "duration":
      raw_weights = torch.clamp(self.motion_lengths, min=1.0e-6)
    else:
      raise ValueError(
        f"Unknown sampling_weight_mode={sampling_weight_mode!r}; "
        "expected 'uniform' or 'duration'."
      )
    self.motion_weights = raw_weights / torch.sum(raw_weights)

    self.total_frames = int(sum(num_frames))
    self.total_duration = float(sum(durations))
    self.max_num_frames = int(max(num_frames))
    self._binary_search_iters = max(
      1, math.ceil(math.log2(self.max_num_frames + 1))
    )

    # Approximate payload only (not PyTorch allocator overhead).
    bytes_per_frame = (
      4 * 1       # timestamp
      + 4 * 3     # left pos
      + 4 * 4     # left quat
      + 4 * 3     # right pos
      + 4 * 4     # right quat
      + 4 * 2     # shoulder mid xy
      + 4 * 1     # left height
      + 4 * 1     # right height
      + 4 * 1     # heading
    )
    approx_mib = self.total_frames * bytes_per_frame / (1024.0**2)

    print(
      f"[teleop] Motion library: {self.num_motions} valid motion(s), "
      f"{self.total_frames} frames, {self.total_duration:.1f}s, "
      f"~{approx_mib:.1f} MiB tensor payload"
    )
    print(f"[teleop] Motion source: {self.source_path}")
    print(
      f"[teleop] Sampling weights: {sampling_weight_mode}; "
      f"recursive_scan={recursive}"
    )

    if invalid:
      print(f"[teleop] Skipped {len(invalid)} invalid NPZ file(s).")
      for path, err in invalid[:5]:
        print(f"[teleop]   skip: {path}: {err}")
      if len(invalid) > 5:
        print(f"[teleop]   ... and {len(invalid) - 5} more")

    if version_counts:
      print(f"[teleop] Interface versions: {version_counts}")
      mismatched = sum(
        count
        for version, count in version_counts.items()
        if version not in (EXPECTED_INTERFACE_VERSION,)
      )
      if mismatched:
        print(
          f"[teleop] WARNING: {mismatched} file(s) do not report "
          f"interface_version={EXPECTED_INTERFACE_VERSION!r}."
        )

  @property
  def num_motions(self) -> int:
    return len(self.motion_files)

  @staticmethod
  def _discover_npz_files(source: Path, recursive: bool) -> list[Path]:
    if source.is_file():
      if source.suffix.lower() != ".npz":
        raise RuntimeError(f"Command file is not an NPZ: {source}")
      return [source]

    if not source.is_dir():
      raise FileNotFoundError(f"Command source does not exist: {source}")

    iterator = source.rglob("*.npz") if recursive else source.glob("*.npz")
    return sorted(
      path
      for path in iterator
      if path.is_file() and not path.name.endswith(".style.npz")
    )

  @classmethod
  def _load_one_npz(
    cls,
    path: Path,
    canonicalize_heading: bool,
    load_style_descriptors: bool,
  ) -> tuple[dict[str, np.ndarray], str]:
    with np.load(path, allow_pickle=False) as data:
      missing = [key for key in cls.REQUIRED_KEYS if key not in data.files]
      if missing:
        raise RuntimeError(f"missing keys {missing}")

      timestamp = np.asarray(data["timestamp"], dtype=np.float64).copy()
      left_pos = np.asarray(
        data["left_wrist_pos_world"], dtype=np.float32
      ).copy()
      left_quat = np.asarray(
        data["left_wrist_quat_world_wxyz"], dtype=np.float32
      ).copy()
      right_pos = np.asarray(
        data["right_wrist_pos_world"], dtype=np.float32
      ).copy()
      right_quat = np.asarray(
        data["right_wrist_quat_world_wxyz"], dtype=np.float32
      ).copy()
      shoulder_mid_xy = np.asarray(
        data["torso_center_xy_world"], dtype=np.float32
      ).copy()
      left_shoulder_h = np.asarray(
        data["left_shoulder_height"], dtype=np.float32
      ).copy()
      right_shoulder_h = np.asarray(
        data["right_shoulder_height"], dtype=np.float32
      ).copy()
      shoulder_heading = np.asarray(
        data["torso_heading_world"], dtype=np.float64
      ).copy()

      metadata = {}
      if "metadata_json" in data.files:
        raw = data["metadata_json"]
        if raw.ndim == 0:
          raw = raw.item()
        try:
          metadata = json.loads(str(raw))
        except Exception:
          metadata = {}

    version = str(metadata.get("interface_version", ""))

    if timestamp.ndim != 1 or len(timestamp) < 2:
      raise RuntimeError("timestamp must contain at least two samples")
    if not np.all(np.isfinite(timestamp)):
      raise RuntimeError("timestamp contains NaN/Inf")
    if np.any(np.diff(timestamp) < 0.0):
      raise RuntimeError("timestamp must be monotonically increasing")

    timestamp -= timestamp[0]
    if float(timestamp[-1]) <= 0.0:
      raise RuntimeError("motion duration must be positive")

    n = len(timestamp)
    expected = {
      "left_wrist_pos_world": (n, 3),
      "left_wrist_quat_world_wxyz": (n, 4),
      "right_wrist_pos_world": (n, 3),
      "right_wrist_quat_world_wxyz": (n, 4),
      "torso_center_xy_world": (n, 2),
      "left_shoulder_height": (n,),
      "right_shoulder_height": (n,),
      "torso_heading_world": (n,),
    }
    actual = {
      "left_wrist_pos_world": left_pos.shape,
      "left_wrist_quat_world_wxyz": left_quat.shape,
      "right_wrist_pos_world": right_pos.shape,
      "right_wrist_quat_world_wxyz": right_quat.shape,
      "torso_center_xy_world": shoulder_mid_xy.shape,
      "left_shoulder_height": left_shoulder_h.shape,
      "right_shoulder_height": right_shoulder_h.shape,
      "torso_heading_world": shoulder_heading.shape,
    }
    for key, shape in expected.items():
      if actual[key] != shape:
        raise RuntimeError(f"{key}: got {actual[key]}, expected {shape}")

    for name, value in (
      ("left_wrist_pos_world", left_pos),
      ("left_wrist_quat_world_wxyz", left_quat),
      ("right_wrist_pos_world", right_pos),
      ("right_wrist_quat_world_wxyz", right_quat),
      ("torso_center_xy_world", shoulder_mid_xy),
      ("left_shoulder_height", left_shoulder_h),
      ("right_shoulder_height", right_shoulder_h),
      ("torso_heading_world", shoulder_heading),
    ):
      if not np.all(np.isfinite(value)):
        raise RuntimeError(f"{name} contains NaN/Inf")

    shoulder_heading = np.unwrap(shoulder_heading)

    if canonicalize_heading:
      heading0 = float(shoulder_heading[0])
      c = np.cos(-heading0)
      s = np.sin(-heading0)

      def rotate_xy_np(xy: np.ndarray) -> np.ndarray:
        out = xy.copy()
        x = xy[..., 0].copy()
        y = xy[..., 1].copy()
        out[..., 0] = c * x - s * y
        out[..., 1] = s * x + c * y
        return out

      left_pos[:, :2] = rotate_xy_np(left_pos[:, :2])
      right_pos[:, :2] = rotate_xy_np(right_pos[:, :2])
      shoulder_mid_xy = rotate_xy_np(shoulder_mid_xy)

      q_delta = np.array(
        [
          np.cos(-0.5 * heading0),
          0.0,
          0.0,
          np.sin(-0.5 * heading0),
        ],
        dtype=np.float32,
      )
      q_delta = np.broadcast_to(q_delta, left_quat.shape)
      left_quat = _np_quat_mul(q_delta, left_quat).astype(np.float32)
      right_quat = _np_quat_mul(q_delta, right_quat).astype(np.float32)
      shoulder_heading -= heading0

    left_quat /= np.maximum(
      np.linalg.norm(left_quat, axis=-1, keepdims=True), 1.0e-8
    )
    right_quat /= np.maximum(
      np.linalg.norm(right_quat, axis=-1, keepdims=True), 1.0e-8
    )

    clip = {
        "timestamp": timestamp.astype(np.float32),
        "left_pos": left_pos,
        "left_quat": left_quat,
        "right_pos": right_pos,
        "right_quat": right_quat,
        "shoulder_mid_xy": shoulder_mid_xy,
        "left_shoulder_h": left_shoulder_h,
        "right_shoulder_h": right_shoulder_h,
        "shoulder_heading": shoulder_heading.astype(np.float32),
    }

    if load_style_descriptors:
      style_path = path.with_name(f"{path.stem}.style.npz")
      if not style_path.is_file():
        raise RuntimeError(f"missing paired descriptor file {style_path.name}")
      with np.load(style_path, allow_pickle=False) as style_data:
        required = (
          "timestamp", "style_descriptor", "style_valid_mask",
          "style_descriptor_names",
        )
        missing = [key for key in required if key not in style_data.files]
        if missing:
          raise RuntimeError(
            f"paired descriptor file {style_path.name} missing keys {missing}"
          )
        style_timestamp = np.asarray(
          style_data["timestamp"], dtype=np.float64
        ).copy()
        descriptor = np.asarray(
          style_data["style_descriptor"], dtype=np.float32
        ).copy()
        valid = np.asarray(
          style_data["style_valid_mask"], dtype=np.bool_
        ).copy()
        names = tuple(str(name) for name in style_data["style_descriptor_names"])

      if style_timestamp.shape != (n,) or not np.allclose(
        style_timestamp - style_timestamp[0], timestamp, rtol=0.0, atol=1.0e-5
      ):
        raise RuntimeError(
          f"paired descriptor timestamps are not synchronized: {style_path.name}"
        )
      if descriptor.shape != (n, len(STYLE_DESCRIPTOR_NAMES)):
        raise RuntimeError(
          f"style_descriptor: got {descriptor.shape}, expected "
          f"{(n, len(STYLE_DESCRIPTOR_NAMES))}"
        )
      if valid.shape != (n,):
        raise RuntimeError(
          f"style_valid_mask: got {valid.shape}, expected {(n,)}"
        )
      if names != STYLE_DESCRIPTOR_NAMES:
        raise RuntimeError("style descriptor schema/order does not match V1 17-D")
      if not np.all(np.isfinite(descriptor)):
        raise RuntimeError("style_descriptor contains NaN/Inf")
      clip["style_descriptor"] = descriptor
      clip["style_valid"] = valid

    return clip, version

  def get_motion_length(self, motion_ids: torch.Tensor) -> torch.Tensor:
    return self.motion_lengths[motion_ids]

  def get_motion_name(self, motion_id: int) -> str:
    return self.motion_names[int(motion_id)]

  def sample_motion_ids(self, n: int) -> torch.Tensor:
    """Sample motion ids independently, with replacement, like TWIST MotionLib."""
    return torch.multinomial(
      self.motion_weights,
      num_samples=n,
      replacement=True,
    )

  def sample_time(self, motion_ids: torch.Tensor) -> torch.Tensor:
    """Uniform random time inside each selected motion."""
    phase = torch.rand(motion_ids.shape, device=self.device)
    return self.motion_lengths[motion_ids] * phase

  def _upper_bound(
    self,
    motion_ids: torch.Tensor,
    t: torch.Tensor,
  ) -> torch.Tensor:
    """Per-env upper_bound(timestamp, t) on ragged flattened arrays."""
    starts = self.motion_start_idx[motion_ids]
    counts = self.motion_num_frames[motion_ids]

    lo = torch.zeros_like(counts)
    hi = counts.clone()  # exclusive

    for _ in range(self._binary_search_iters):
      active = lo < hi
      mid = torch.div(lo + hi, 2, rounding_mode="floor")
      safe_mid = torch.minimum(mid, counts - 1)
      ts = self.timestamp_flat[starts + safe_mid]

      # upper_bound: first timestamp > t
      go_right = active & (ts <= t)
      lo = torch.where(go_right, mid + 1, lo)
      hi = torch.where(active & ~go_right, mid, hi)

    return lo

  def sample(
    self,
    motion_ids: torch.Tensor,
    query_time: torch.Tensor,
  ) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
  ]:
    """Vectorized timestamp interpolation for independently selected motions."""
    if motion_ids.shape != query_time.shape:
      raise ValueError(
        f"motion_ids/query_time shape mismatch: "
        f"{motion_ids.shape} vs {query_time.shape}"
      )

    i0, i1, alpha = self._sample_indices(motion_ids, query_time)

    def lerp(x: torch.Tensor) -> torch.Tensor:
      x0 = x[i0]
      x1 = x[i1]
      a = alpha
      while a.ndim < x0.ndim:
        a = a.unsqueeze(-1)
      return x0 + a * (x1 - x0)

    return (
      lerp(self.npz_left_wrist_pos_cw),
      _quat_nlerp(
        self.npz_left_wrist_quat_cw[i0],
        self.npz_left_wrist_quat_cw[i1],
        alpha,
      ),
      lerp(self.npz_right_wrist_pos_cw),
      _quat_nlerp(
        self.npz_right_wrist_quat_cw[i0],
        self.npz_right_wrist_quat_cw[i1],
        alpha,
      ),
      lerp(self.npz_shoulder_mid_xy_cw),
      lerp(self.npz_left_shoulder_height),
      lerp(self.npz_right_shoulder_height),
      lerp(self.npz_shoulder_heading_cw),
    )

  def _sample_indices(
    self,
    motion_ids: torch.Tensor,
    query_time: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if motion_ids.shape != query_time.shape:
      raise ValueError(
        f"motion_ids/query_time shape mismatch: "
        f"{motion_ids.shape} vs {query_time.shape}"
      )
    lengths = self.motion_lengths[motion_ids]
    t = torch.minimum(torch.clamp(query_time, min=0.0), lengths)
    starts = self.motion_start_idx[motion_ids]
    counts = self.motion_num_frames[motion_ids]
    upper = self._upper_bound(motion_ids, t)
    local_i1 = torch.minimum(torch.clamp(upper, min=1), counts - 1)
    local_i0 = local_i1 - 1
    i0 = starts + local_i0
    i1 = starts + local_i1
    t0 = self.timestamp_flat[i0]
    t1 = self.timestamp_flat[i1]
    alpha = (t - t0) / torch.clamp(t1 - t0, min=1.0e-6)
    return i0, i1, torch.clamp(alpha, min=0.0, max=1.0)

  def sample_style(
    self,
    motion_ids: torch.Tensor,
    query_time: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate synchronized human descriptors at command query times."""
    if self.style_descriptor_flat is None or self.style_valid_flat is None:
      raise RuntimeError("Style descriptors were not loaded for this library.")
    i0, i1, alpha = self._sample_indices(motion_ids, query_time)
    descriptor = (
      self.style_descriptor_flat[i0] * (1.0 - alpha[:, None])
      + self.style_descriptor_flat[i1] * alpha[:, None]
    )
    valid = self.style_valid_flat[i0] & self.style_valid_flat[i1]
    return descriptor, valid
