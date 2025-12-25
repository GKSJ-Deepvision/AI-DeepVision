import smtplib, ssl
from email.message import EmailMessage
from datetime import datetime
import os

EMAIL = "Email ID"
APP_PASSWORD = "Password"
PORT = 465  # Already defined

def send_email(to_email, count, alert_text, fps, snapshot_path=None):
    """
    Sends a crowd alert email including timestamp, count, alert, FPS, and optional snapshot.

    Parameters:
    - to_email (str): recipient email
    - count (int): current crowd count
    - alert_text (str): description of the alert
    - fps (float): current FPS
    - snapshot_path (str): optional path to snapshot image that triggered alert
    """
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Compose email body
    body = f"""
Crowd Alert Observed \n
Check Details here: 
Time: {now}
Count: {count}
Alert: {alert_text}
FPS: {fps:.2f}
"""
    
    msg = EmailMessage()
    msg["Subject"] = "🚨 Crowd Monitoring Alert"
    msg["From"] = EMAIL
    msg["To"] = to_email
    msg.set_content(body)
    
    # Attach snapshot if provided
    if snapshot_path and os.path.exists(snapshot_path):
        with open(snapshot_path, "rb") as f:
            img_data = f.read()
            import imghdr
            img_type = imghdr.what(f.name)
            msg.add_attachment(img_data, maintype="image", subtype=img_type, filename=os.path.basename(snapshot_path))
    
    # Send email via SSL
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", PORT, context=context) as server:
        server.login(EMAIL, APP_PASSWORD)
        server.send_message(msg)
    
    print(f"Alert email sent to {to_email} at {now}")