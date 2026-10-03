from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from .motion_library import STYLE_DESCRIPTOR_NAMES, SparseWholeBodyMotionLibrary


KNEEL_SIDE_ENV_VAR = "UNITREE_TELEOP_KNEEL_SIDE"


class KneelSparseWholeBodyMotionLibrary(SparseWholeBodyMotionLibrary):
  """Compatibility layer for the new EgoDex+PICO synthesized dataset.

  The base teleop motion library remains unchanged.  This subclass adds three
  pieces of behavior needed by the static-kneel experiments:

  1) ``*.style_candidates.npz`` is accepted as a style source.  In descriptor
     mode we select the candidate recorded in the command NPZ metadata
     (``selected_pico_template_index``), preserving one coherent human support
     strategy for the whole episode.
  2) ``phase`` and ``wrist_tracking_weight_scale`` are loaded as optional
     command annotations.  Older command NPZs default to phase=0 and weight=1.
  3) ``UNITREE_TELEOP_KNEEL_SIDE={any,left,right}`` can filter synthesized
     episodes by ``selected_pico_side`` without physically duplicating the
     dataset tree.

  This intentionally does *not* expose the selected human descriptor to the
  actor.  The descriptor remains reward-only, matching the existing teacher
  design.
  """

  @staticmethod
  def _discover_npz_files(source: Path, recursive: bool) -> list[Path]:
    files = SparseWholeBodyMotionLibrary._discover_npz_files(source, recursive)
    return [
      p for p in files
      if not p.name.endswith(".style_candidates.npz")
      and p.name != "kneel_template_library.npz"
    ]

  @staticmethod
  def _read_command_metadata(path: Path) -> dict:
    try:
      with np.load(path, allow_pickle=False) as data:
        if "metadata_json" not in data.files:
          return {}
        raw = data["metadata_json"]
        if raw.ndim == 0:
          raw = raw.item()
        return json.loads(str(raw))
    except Exception:
      return {}

  @classmethod
  def _load_one_npz(
    cls,
    path: Path,
    canonicalize_heading: bool,
    load_style_descriptors: bool,
  ) -> tuple[dict[str, np.ndarray], str]:
    metadata = cls._read_command_metadata(path)

    side_filter = os.environ.get(KNEEL_SIDE_ENV_VAR, "any").strip().lower()
    if side_filter not in ("", "any", "left", "right"):
      raise RuntimeError(
        f"{KNEEL_SIDE_ENV_VAR} must be any/left/right, got {side_filter!r}"
      )
    selected_side = str(metadata.get("selected_pico_side", "")).lower()
    if side_filter in ("left", "right") and selected_side != side_filter:
      raise RuntimeError(
        f"filtered by {KNEEL_SIDE_ENV_VAR}={side_filter}: "
        f"selected_pico_side={selected_side or 'missing'}"
      )

    # Legacy descriptor datasets still go through the stock path unchanged.
    if not load_style_descriptors:
      return super()._load_one_npz(
        path,
        canonicalize_heading=canonicalize_heading,
        load_style_descriptors=False,
      )

    legacy_style = path.with_name(f"{path.stem}.style.npz")
    if legacy_style.is_file():
      return super()._load_one_npz(
        path,
        canonicalize_heading=canonicalize_heading,
        load_style_descriptors=True,
      )

    # New synthesis format: select one coherent candidate for the entire clip.
    clip, version = super()._load_one_npz(
      path,
      canonicalize_heading=canonicalize_heading,
      load_style_descriptors=False,
    )
    style_path = path.with_name(f"{path.stem}.style_candidates.npz")
    if not style_path.is_file():
      raise RuntimeError(
        f"missing paired descriptor file {style_path.name} "
        f"(or legacy {legacy_style.name})"
      )

    with np.load(style_path, allow_pickle=False) as style_data:
      required = (
        "timestamp",
        "style_candidates",
        "style_candidate_valid",
        "style_descriptor_names",
      )
      missing = [key for key in required if key not in style_data.files]
      if missing:
        raise RuntimeError(
          f"paired candidate file {style_path.name} missing keys {missing}"
        )

      style_timestamp = np.asarray(
        style_data["timestamp"], dtype=np.float64
      ).copy()
      candidates = np.asarray(
        style_data["style_candidates"], dtype=np.float32
      ).copy()
      valid = np.asarray(
        style_data["style_candidate_valid"], dtype=np.bool_
      ).copy()
      names = tuple(
        str(name) for name in style_data["style_descriptor_names"]
      )
      sides = None
      if "style_candidate_sides" in style_data.files:
        sides = np.asarray(style_data["style_candidate_sides"]).astype(str)

    timestamp = np.asarray(clip["timestamp"], dtype=np.float64)
    n = len(timestamp)
    if style_timestamp.shape != (n,) or not np.allclose(
      style_timestamp - style_timestamp[0],
      timestamp - timestamp[0],
      rtol=0.0,
      atol=1.0e-5,
    ):
      raise RuntimeError(
        f"paired candidate timestamps are not synchronized: {style_path.name}"
      )
    if candidates.ndim != 3 or candidates.shape[1:] != (
      n,
      len(STYLE_DESCRIPTOR_NAMES),
    ):
      raise RuntimeError(
        f"style_candidates: got {candidates.shape}, expected "
        f"(K,{n},{len(STYLE_DESCRIPTOR_NAMES)})"
      )
    if valid.shape != candidates.shape[:2]:
      raise RuntimeError(
        f"style_candidate_valid: got {valid.shape}, expected "
        f"{candidates.shape[:2]}"
      )
    if names != STYLE_DESCRIPTOR_NAMES:
      raise RuntimeError("style descriptor schema/order does not match V1 17-D")
    if not np.all(np.isfinite(candidates)):
      raise RuntimeError("style_candidates contains NaN/Inf")

    candidate_index = metadata.get("selected_pico_template_index", None)
    try:
      candidate_index = int(candidate_index)
    except (TypeError, ValueError):
      candidate_index = -1

    if not 0 <= candidate_index < candidates.shape[0]:
      # Fall back to the first candidate of the recorded side when possible.
      candidate_index = 0
      if sides is not None and selected_side in ("left", "right"):
        matches = np.flatnonzero(np.char.lower(sides) == selected_side)
        if len(matches):
          candidate_index = int(matches[0])

    clip["style_descriptor"] = candidates[candidate_index]
    clip["style_valid"] = valid[candidate_index]
    return clip, version

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)

    phase_list: list[np.ndarray] = []
    task_weight_list: list[np.ndarray] = []
    for path, count_t in zip(self.motion_files, self.motion_num_frames):
      count = int(count_t.item())
      phase = np.zeros(count, dtype=np.int64)
      task_weight = np.ones(count, dtype=np.float32)
      with np.load(path, allow_pickle=False) as data:
        if "phase" in data.files:
          value = np.asarray(data["phase"], dtype=np.int64)
          if value.shape != (count,):
            raise RuntimeError(
              f"{path}: phase shape {value.shape}, expected {(count,)}"
            )
          phase = value
        if "wrist_tracking_weight_scale" in data.files:
          value = np.asarray(
            data["wrist_tracking_weight_scale"], dtype=np.float32
          )
          if value.shape != (count,):
            raise RuntimeError(
              f"{path}: wrist_tracking_weight_scale shape {value.shape}, "
              f"expected {(count,)}"
            )
          if not np.all(np.isfinite(value)) or np.any(value < 0.0):
            raise RuntimeError(
              f"{path}: wrist_tracking_weight_scale must be finite and >= 0"
            )
          task_weight = value
      phase_list.append(phase)
      task_weight_list.append(task_weight)

    self.phase_flat = torch.as_tensor(
      np.concatenate(phase_list), dtype=torch.long, device=self.device
    )
    self.wrist_tracking_weight_flat = torch.as_tensor(
      np.concatenate(task_weight_list),
      dtype=torch.float32,
      device=self.device,
    )

  def sample_aux(
    self,
    motion_ids: torch.Tensor,
    query_time: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample discrete phase and interpolated wrist-task weight."""
    i0, i1, alpha = self._sample_indices(motion_ids, query_time)
    phase = torch.where(
      alpha < 0.5,
      self.phase_flat[i0],
      self.phase_flat[i1],
    )
    weight = (
      self.wrist_tracking_weight_flat[i0] * (1.0 - alpha)
      + self.wrist_tracking_weight_flat[i1] * alpha
    )
    return phase, weight.clamp_min(0.0)
