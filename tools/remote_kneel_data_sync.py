#!/usr/bin/env python3
"""Synchronize and verify the complete EgoDex-PICO kneeling dataset."""

from __future__ import annotations

import argparse
import hashlib
import shlex
import subprocess
import tempfile
from pathlib import Path


DEFAULT_SOURCE = Path(
  "/mnt/hdd/humanoid_locomotion/datasets/egodex_pico_kneel_synthesis/"
  "EgoDex-PICO-kneel-synth"
)
REMOTE_HOST = "unitree-trainer"
REMOTE_DIR = "/home/dev/EgoDex-PICO-kneel-synth"


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()
  source = args.source.expanduser().resolve()
  if not source.is_dir():
    raise FileNotFoundError(source)
  files = sorted(path for path in source.rglob("*") if path.is_file())
  if not files:
    raise RuntimeError(f"No dataset files in {source}")
  total_bytes = sum(path.stat().st_size for path in files)
  print(f"dataset files={len(files)}, bytes={total_bytes}", flush=True)
  print(f"remote destination={REMOTE_HOST}:{REMOTE_DIR}", flush=True)
  if args.dry_run:
    return

  with tempfile.TemporaryDirectory() as temp:
    checksums = Path(temp) / "SHA256SUMS"
    with checksums.open("w") as output:
      for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
          for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        output.write(f"{digest.hexdigest()}  {path.relative_to(source)}\n")
    subprocess.run(
      ["ssh", REMOTE_HOST, f"mkdir -p -- {shlex.quote(REMOTE_DIR)}"],
      check=True,
    )
    subprocess.run(
      [
        "rsync", "-rlt", "--omit-dir-times", "--quiet",
        str(source) + "/", f"{REMOTE_HOST}:{REMOTE_DIR}/",
      ],
      check=True,
    )
    subprocess.run(
      ["scp", "-q", str(checksums), f"{REMOTE_HOST}:{REMOTE_DIR}/SHA256SUMS"],
      check=True,
    )
    subprocess.run(
      [
        "ssh", REMOTE_HOST,
        f"cd {shlex.quote(REMOTE_DIR)} && sha256sum -c SHA256SUMS",
      ],
      check=True,
      stdout=subprocess.DEVNULL,
    )
  print("Remote dataset checksum verification passed.")


if __name__ == "__main__":
  main()
