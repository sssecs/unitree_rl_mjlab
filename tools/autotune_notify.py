#!/usr/bin/env python3
"""Send an optional SMTP notification without third-party dependencies."""

from __future__ import annotations

import argparse
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path


def load_env(path: Path) -> dict[str, str]:
  values = dict(os.environ)
  if not path.is_file():
    return values
  for raw_line in path.read_text().splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
      continue
    key, value = line.split("=", 1)
    values.setdefault(key.strip(), value.strip().strip("\"'"))
  return values


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("subject")
  parser.add_argument("body_file", type=Path)
  parser.add_argument("--config", type=Path, default=Path(".autotune/notify.env"))
  args = parser.parse_args()
  cfg = load_env(args.config)
  required = (
    "AUTOTUNE_SMTP_HOST",
    "AUTOTUNE_SMTP_USERNAME",
    "AUTOTUNE_SMTP_PASSWORD",
    "AUTOTUNE_EMAIL_FROM",
    "AUTOTUNE_EMAIL_TO",
  )
  missing = [key for key in required if not cfg.get(key)]
  if missing:
    print("[autotune] email disabled; missing " + ", ".join(missing))
    return

  message = EmailMessage()
  message["Subject"] = args.subject
  message["From"] = cfg["AUTOTUNE_EMAIL_FROM"]
  message["To"] = cfg["AUTOTUNE_EMAIL_TO"]
  message.set_content(args.body_file.read_text())
  host = cfg["AUTOTUNE_SMTP_HOST"]
  port = int(cfg.get("AUTOTUNE_SMTP_PORT", "465"))
  use_ssl = cfg.get("AUTOTUNE_SMTP_SSL", "true").lower() == "true"
  context = ssl.create_default_context()
  if use_ssl:
    smtp = smtplib.SMTP_SSL(host, port, timeout=30, context=context)
  else:
    smtp = smtplib.SMTP(host, port, timeout=30)
    smtp.starttls(context=context)
  with smtp:
    smtp.login(cfg["AUTOTUNE_SMTP_USERNAME"], cfg["AUTOTUNE_SMTP_PASSWORD"])
    smtp.send_message(message)
  print(f"[autotune] stop email sent to {cfg['AUTOTUNE_EMAIL_TO']}")


if __name__ == "__main__":
  main()
