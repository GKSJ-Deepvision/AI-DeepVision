import smtplib, ssl
from email.message import EmailMessage
from datetime import datetime
import os

EMAIL = "receiver@gmail.com"          # 🔴 change this
APP_PASSWORD = "receiverpassword"     # 🔴 Gmail App Password
PORT = 465

def send_email(to_email, count, alert_text, fps, snapshot_path=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body = f"""
Crowd Alert Detected

Time: {now}
Crowd Count: {count}
Alert Status: {alert_text}
FPS: {fps:.2f}
"""

    msg = EmailMessage()
    msg["Subject"] = "🚨 DeepVision Crowd Alert"
    msg["From"] = EMAIL
    msg["To"] = to_email
    msg.set_content(body)

    if snapshot_path and os.path.exists(snapshot_path):
        with open(snapshot_path, "rb") as f:
            img_data = f.read()
            import imghdr
            img_type = imghdr.what(f.name)
            msg.add_attachment(
                img_data,
                maintype="image",
                subtype=img_type,
                filename=os.path.basename(snapshot_path)
            )

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", PORT, context=context) as server:
        server.login(EMAIL, APP_PASSWORD)
        server.send_message(msg)

    print(f"✅ Alert email sent to {to_email} at {now}")
