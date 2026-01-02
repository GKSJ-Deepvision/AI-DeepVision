import smtplib
from email.message import EmailMessage
from datetime import datetime

# Simple in-memory list of emails (no DB, easier)
EMAIL_LIST = []

SENDER_EMAIL = "aksharasrik14@gmail.com"       # put your sender
APP_PASSWORD = "lqfw zxvf loaf mxxs"      # Gmail app password

def add_email(email: str):
    if email and email not in EMAIL_LIST:
        EMAIL_LIST.append(email)

def get_emails():
    return EMAIL_LIST

def send_alert(to_email: str, count: float):
    now = datetime.now()
    msg = EmailMessage()
    msg["Subject"] = "🚨 Crowd Density Alert"
    msg["From"] = SENDER_EMAIL
    msg["To"] = to_email
    msg.set_content(
        f"Crowd alert at {now:%Y-%m-%d %H:%M:%S}\nEstimated crowd count: {int(count)}"
    )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(SENDER_EMAIL, APP_PASSWORD)
        smtp.send_message(msg)
