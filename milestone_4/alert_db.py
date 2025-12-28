import sqlite3

def init_db():
    conn = sqlite3.connect("alert_emails.db")
    conn.execute("CREATE TABLE IF NOT EXISTS emails (email TEXT)")
    conn.close()

def add_email(email):
    conn = sqlite3.connect("alert_emails.db")
    conn.execute("INSERT INTO emails VALUES (?)", (email,))
    conn.commit()
    conn.close()

def get_emails():
    conn = sqlite3.connect("alert_emails.db")
    emails = [e[0] for e in conn.execute("SELECT email FROM emails")]
    conn.close()
    return emails
