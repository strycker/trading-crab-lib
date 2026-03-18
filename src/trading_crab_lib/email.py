"""
Weekly email delivery for market regime reports.

SMTP config is loaded from a path provided by the caller (no default path).
Accepts both key schemas: sender/recipients and from_address/to_address.
"""

from __future__ import annotations

import logging
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


def load_email_config(config_path: Path | None = None) -> dict:
    """
    Load email configuration from a YAML file.

    Args:
        config_path: Path to email YAML. If None, returns {} (no file read).

    Expected keys (either schema):
        sender / from_address   — sender email
        recipients / to_address — recipient(s); to_address can be str or list
        smtp_host, smtp_port, username, password
        use_ssl (optional)

    Returns empty dict if path is None, missing, or malformed.
    """
    if config_path is None or not config_path.exists():
        if config_path is not None:
            log.warning("Email config not found at %s", config_path)
        return {}

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            log.warning("Email config is not a dict: %s", config_path)
            return {}
        return _normalize_email_config(cfg)
    except Exception as exc:
        log.warning("Failed to load email config: %s", exc)
        return {}


def _normalize_email_config(cfg: dict) -> dict:
    """Ensure config has sender and recipients (from from_address/to_address if present)."""
    out = dict(cfg)
    if out.get("sender") is None and out.get("from_address") is not None:
        out["sender"] = out["from_address"]
    if out.get("recipients") is None:
        if out.get("to_address") is not None:
            to = out["to_address"]
            out["recipients"] = [to] if isinstance(to, str) else list(to)
        elif out.get("to_addresses") is not None:
            to = out["to_addresses"]
            out["recipients"] = [to] if isinstance(to, str) else list(to)
    return out


def build_weekly_email_body(
    report_dir: Path,
    subject_prefix: str = "Market Regime Weekly Report",
) -> tuple[str, str]:
    """
    Build email subject and body from the most recent report files.

    Looks for (in order):
      1. report_dir/email_body.txt  — pre-formatted email body
      2. report_dir/weekly_report.md — markdown report (used as-is)
      3. report_dir/dashboard.csv — fallback: summarize CSV

    Returns:
        (subject, body) tuple.
    """
    from datetime import date
    subject = f"{subject_prefix} — {date.today().isoformat()}"

    email_body_file = report_dir / "email_body.txt"
    if email_body_file.exists():
        body = email_body_file.read_text(encoding="utf-8")
        return subject, body

    report_file = report_dir / "weekly_report.md"
    if report_file.exists():
        body = report_file.read_text(encoding="utf-8")
        return subject, body

    dashboard_file = report_dir / "dashboard.csv"
    if dashboard_file.exists():
        import pandas as pd
        df = pd.read_csv(dashboard_file)
        body = f"Dashboard summary ({len(df)} assets):\n\n{df.to_string(index=False)}"
        return subject, body

    return subject, "(No report files found)"


def send_weekly_email(
    config: dict,
    subject: str,
    body: str,
) -> bool:
    """
    Send an email via SMTP.

    Config may use either schema: sender/recipients or from_address/to_address
    (both are normalized internally).

    Args:
        config:  dict from load_email_config() or with sender, recipients, etc.
        subject: email subject line
        body:    email body (plain text)

    Returns:
        True if sent successfully, False otherwise.
    """
    cfg = _normalize_email_config(config)
    required = ["smtp_host", "smtp_port", "username", "password", "sender", "recipients"]
    missing = [k for k in required if k not in cfg or not cfg[k]]
    if missing:
        log.error("Email config missing required keys: %s", missing)
        return False

    recipients = cfg["recipients"]
    if isinstance(recipients, str):
        recipients = [recipients]

    if not recipients:
        log.error("No recipients configured")
        return False

    msg = MIMEMultipart()
    msg["From"] = cfg["sender"]
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    use_ssl = cfg.get("use_ssl", False)
    host = cfg["smtp_host"]
    port = int(cfg["smtp_port"])

    try:
        if use_ssl:
            context = ssl.create_default_context()
            server = smtplib.SMTP_SSL(host, port, context=context)
        else:
            server = smtplib.SMTP(host, port)
            server.starttls()

        server.login(cfg["username"], cfg["password"])
        server.sendmail(cfg["sender"], recipients, msg.as_string())
        server.quit()
        log.info("Weekly email sent to %s", recipients)
        return True
    except Exception as exc:
        log.error("Failed to send email: %s", exc)
        return False
