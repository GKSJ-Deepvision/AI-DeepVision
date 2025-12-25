import sqlite3

DB_PATH = "emails.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS emails (addr TEXT)")
    return conn

def add_email(addr):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO emails VALUES (?)", (addr,))
    conn.commit()
    conn.close()

def get_emails():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT addr FROM emails")
    emails = [e[0] for e in cur.fetchall()]
    conn.close()
    return emails
