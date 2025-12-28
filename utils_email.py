import sqlite3
import smtplib
import os
from email.mime.text import MIMEText
from dotenv import load_dotenv

# ---------- LOAD ENV ----------
load_dotenv()

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

SENDER_EMAIL = os.getenv("MAIL_ID")
SENDER_PASSWORD = os.getenv("MAIL_PASS")

DB_PATH = "emails.db"

# ---------- DB ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS emails (
            email TEXT UNIQUE
        )
    """)
    conn.commit()
    conn.close()


def get_all_emails():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT email FROM emails")
    emails = [row[0] for row in c.fetchall()]
    conn.close()
    return emails


# ---------- EMAIL ----------
def send_alert_emails(emails, density_index):
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        return "❌ Email credentials not loaded (.env error)"

    if not emails:
        return "❌ No alert emails in database"

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)

        for email in emails:
            msg = MIMEText(f"""
🚨 CROWD DENSITY ALERT 🚨

Detected Density Index: {density_index:.2f}

Immediate action recommended.

– Crowd Monitoring System (CSRNet)
""")
            msg["Subject"] = "Crowd Density Alert"
            msg["From"] = SENDER_EMAIL
            msg["To"] = email

            server.sendmail(SENDER_EMAIL, email, msg.as_string())

        server.quit()
        return "✅ Alert email sent successfully"

    except Exception as e:
        return f"❌ SMTP Error: {e}"
