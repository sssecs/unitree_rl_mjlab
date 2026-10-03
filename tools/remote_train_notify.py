#!/usr/bin/env python3
"""Email the owner when one remote training session exits."""

from __future__ import annotations

import argparse
import os
import smtplib
import socket
import ssl
import sys
from email.message import EmailMessage
from pathlib import Path


REQUIRED = (
  "AUTOTUNE_SMTP_HOST",
  "AUTOTUNE_SMTP_USERNAME",
  "AUTOTUNE_SMTP_PASSWORD",
  "AUTOTUNE_EMAIL_FROM",
  "AUTOTUNE_EMAIL_TO",
)


def load_config(path: Path) -> dict[str, str]:
  config = dict(os.environ)
  for raw in path.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
      continue
    key, value = line.split("=", 1)
    config.setdefault(key.strip(), value.strip().strip("\"'"))
  missing = [key for key in REQUIRED if not config.get(key)]
  if missing:
    raise ValueError("missing notification settings: " + ", ".join(missing))
  return config


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--session", required=True)
  parser.add_argument("--exit-code", required=True, type=int)
  parser.add_argument("--run-dir", required=True, type=Path)
  parser.add_argument("--config", required=True, type=Path)
  args = parser.parse_args()

  try:
    cfg = load_config(args.config)
    status = "completed" if args.exit_code == 0 else "failed"
    log_dir_file = args.run_dir / "training_log_dir"
    log_dir = log_dir_file.read_text().strip() if log_dir_file.is_file() else "unknown"
    started_file = args.run_dir / "started_at"
    started = started_file.read_text().strip() if started_file.is_file() else "unknown"
    ended_file = args.run_dir / "ended_at"
    ended = ended_file.read_text().strip() if ended_file.is_file() else "unknown"

    message = EmailMessage()
    message["Subject"] = f"G1 training {status}: {args.session}"
    message["From"] = cfg["AUTOTUNE_EMAIL_FROM"]
    message["To"] = cfg["AUTOTUNE_EMAIL_TO"]
    message.set_content(
      f"Session: {args.session}\n"
      f"Status: {status}\n"
      f"Exit code: {args.exit_code}\n"
      f"Host: {socket.gethostname()}\n"
      f"Started: {started}\n"
      f"Ended: {ended}\n"
      f"Training log directory: {log_dir}\n"
      f"Run record: {args.run_dir}\n"
    )

    host = cfg["AUTOTUNE_SMTP_HOST"]
    port = int(cfg.get("AUTOTUNE_SMTP_PORT", "465"))
    context = ssl.create_default_context()
    if cfg.get("AUTOTUNE_SMTP_SSL", "true").lower() == "true":
      smtp = smtplib.SMTP_SSL(host, port, timeout=30, context=context)
    else:
      smtp = smtplib.SMTP(host, port, timeout=30)
      smtp.starttls(context=context)
    with smtp:
      smtp.login(cfg["AUTOTUNE_SMTP_USERNAME"], cfg["AUTOTUNE_SMTP_PASSWORD"])
      smtp.send_message(message)
    print("Training completion email sent.")
    return 0
  except (OSError, ValueError, smtplib.SMTPException) as exc:
    # Keep credentials and server replies out of the training record.
    print(f"Training completion email failed: {type(exc).__name__}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
