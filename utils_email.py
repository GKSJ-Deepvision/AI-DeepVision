import sqlite3
import smtplib
import os
from email.mime.text import MIMEText
from dotenv import load_dotenv

# =====================================================
# LOAD ENV (CORRECT WAY)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH, override=True)

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

SENDER_EMAIL = os.getenv("MAIL_ID")
SENDER_PASSWORD = os.getenv("MAIL_PASS")

# ---- DEBUG (REMOVE AFTER IT WORKS) ----
print("📧 MAIL_ID =", SENDER_EMAIL)
print("🔐 MAIL_PASS loaded =", bool(SENDER_PASSWORD))

# =====================================================
# DATABASE
# =====================================================
DB_PATH = os.path.join(BASE_DIR, "emails.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS emails (
            email TEXT UNIQUE NOT NULL
        )
    """)

    # 🔥 AUTO-INSERT ADMIN EMAIL (CLOUD SAFE)
    cursor.execute(
        "INSERT OR IGNORE INTO emails (email) VALUES (?)",
        ("anvithasateesh1408@gmail.com",)
    )

    conn.commit()
    conn.close()



def get_all_emails():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT email FROM emails")
    emails = [row[0] for row in cursor.fetchall()]
    conn.close()
    return emails


# =====================================================
# EMAIL SENDER
# =====================================================
def send_alert_emails(emails, density_index):
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        return "❌ Email credentials not loaded. Check .env file."

    if not emails:
        return "❌ No alert emails found in database"

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)

        for email in emails:
            message_body = f"""\
🚨 CROWD DENSITY ALERT 🚨

Detected Density Index: {density_index:.2f}

Immediate action recommended.

– Crowd Monitoring System (CSRNet)
"""

            msg = MIMEText(message_body)
            msg["Subject"] = "Crowd Density Alert"
            msg["From"] = SENDER_EMAIL
            msg["To"] = email

            server.sendmail(SENDER_EMAIL, email, msg.as_string())

        server.quit()
        return "✅ Alert email(s) sent successfully"

    except Exception as e:
        return f"❌ SMTP Error: {str(e)}"
