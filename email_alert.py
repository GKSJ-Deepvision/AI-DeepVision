import os
import smtplib
import logging
from email.mime.text import MIMEText
from typing import List

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _get_config():
    """Return current email configuration read from environment variables."""
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")

    try:
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
    except ValueError:
        smtp_port = 587

    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    receivers = [
        s.strip()
        for s in os.getenv("EMAIL_RECEIVERS", "").split(",")
        if s.strip()
    ]

    return smtp_server, smtp_port, sender, password, receivers


def send_email_alert(count: int) -> bool:
    """Send a crowd alert email. Returns True on success, False on failure."""
    smtp_server, smtp_port, sender, password, receivers = _get_config()

    if not (sender and password and receivers):
        log.error("Email not sent: missing EMAIL_SENDER, EMAIL_PASSWORD, or EMAIL_RECEIVERS")
        return False

    msg = MIMEText(
        f"⚠ Crowd Alert!\n\nDetected crowd count: {count}\n\nAI-DeepVision System"
    )
    msg["Subject"] = "🚨 Crowd Alert Notification"
    msg["From"] = sender
    msg["To"] = ", ".join(receivers)

    try:
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=10)
        server.ehlo()
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()

        log.info("Alert email sent to: %s", receivers)
        return True

    except Exception as exc:
        log.exception("Failed to send alert email: %s", exc)
        return False


def send_test_email() -> bool:
    """Send a simple test email to configured receivers."""
    smtp_server, smtp_port, sender, password, receivers = _get_config()

    if not (sender and password and receivers):
        log.warning("Test email not sent: email not configured")
        return False

    msg = MIMEText(
        "This is a test email from AI-DeepVision.\n\n"
        "If you received this, the SMTP configuration is working correctly."
    )
    msg["Subject"] = "AI-DeepVision Test Email"
    msg["From"] = sender
    msg["To"] = ", ".join(receivers)

    return _send_message(msg)


def _send_message(msg: MIMEText) -> bool:
    """Internal helper to send a MIMEText message object."""
    smtp_server, smtp_port, sender, password, receivers = _get_config()

    if not (sender and password and receivers):
        log.error("Email not sent: email not configured")
        return False

    try:
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=10)
        server.ehlo()
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()

        log.info("Email sent successfully")
        return True

    except Exception as exc:
        log.exception("Failed to send email: %s", exc)
        return False


def is_email_configured() -> bool:
    """Return True if email environment variables are configured."""
    _, _, sender, password, receivers = _get_config()
    return bool(sender and password and receivers)


def get_email_receivers() -> List[str]:
    """Return the configured receiver email addresses."""
    _, _, _, _, receivers = _get_config()
    return receivers
