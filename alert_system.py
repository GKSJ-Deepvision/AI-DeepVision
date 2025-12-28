import smtplib
from email.mime.text import MIMEText

#ALERT_THRESHOLD = 4

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "artikumari000009@gmail.com"
SENDER_PASSWORD = "vhpv fkdu uebd glao"
RECIPIENTS = ["a73066923@gmail.com"]

def send_email_alert(count, mode):
    try:
        msg = MIMEText(
            f"Crowd Alert!\n\nCount: {count}\nMode: {mode}"
        )
        msg["Subject"] = "🚨 Crowd Threshold Exceeded"
        msg["From"] = SENDER_EMAIL
        msg["To"] = ", ".join(RECIPIENTS)

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()

        print("✅ Alert email sent successfully")

    except Exception as e:
        print("❌ Email failed:", e)


