AI-DeepVision

This project provides a Streamlit dashboard for crowd monitoring using CSRNet.

Email alert configuration

1. Create an app password (Gmail) if your account has 2FA enabled. Use that as `EMAIL_PASSWORD`.
2. Copy `.env.example` to `.env` and update the values.
3. Required env vars:
   - `EMAIL_SENDER` (your email)
   - `EMAIL_PASSWORD` (app password)
   - `EMAIL_RECEIVERS` (comma-separated recipient emails)
   - `SMTP_SERVER` (defaults to `smtp.gmail.com`)
   - `SMTP_PORT` (defaults to `587`)

Local testing

- In development, you can install `python-dotenv` and load env vars from `.env` before running Streamlit:

```py
from dotenv import load_dotenv
load_dotenv()
```

Then run:

```
streamlit run app.py
```

Logging

- Email send failures are logged. Check the terminal output for error details.
