import smtplib
from email.message import EmailMessage
from datetime import datetime

# ---------------- EMAIL CONFIG ----------------
SENDER_EMAIL = "mahalakshmitetala0909@gmail.com"
APP_PASSWORD = "vftkknwqeldegiiy"  # Gmail App Password

# ---------------- SEND ALERT ----------------
def send_alert(to_email, count):
    """
    Sends crowd alert email with separate date and time
    """

    # Get current date and time
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Create email
    msg = EmailMessage()
    msg["Subject"] = "🚨 Crowd Density Alert"
    msg["From"] = SENDER_EMAIL
    msg["To"] = to_email

    msg.set_content(
        f"""
Crowd Alert Detected

Date: {date_str}
Time: {time_str}
Estimated Crowd Count: {int(count)}
"""
    )

    # Send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)
