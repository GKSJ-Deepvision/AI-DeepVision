import streamlit as st
import cv2
import torch
import numpy as np
import logging

# Optional: load environment variables from a .env file in development
loaded_dotenv = False
try:
    from dotenv import load_dotenv
    load_dotenv()
    loaded_dotenv = True
except Exception:
    # Fallback: if python-dotenv is not installed, try a minimal .env loader
    import os
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                # Don't overwrite existing env vars
                if key not in os.environ:
                    os.environ[key] = val
        loaded_dotenv = True

logging.basicConfig(level=logging.INFO)

from utils.csrnet_model import CSRNet
from utils.preprocess import preprocess_frame
from utils.heatmap import generate_heatmap
from email_alert import send_email_alert, send_test_email, is_email_configured, get_email_receivers

# ---------------- CONFIG ----------------
MODEL_PATH = "models/best_csrnet_partB.pth"
ALERT_THRESHOLD = 150

st.set_page_config(page_title="AI-DeepVision", layout="wide")
st.title("AI-DeepVision | Crowd Monitoring Dashboard")

# Show email configuration status in the sidebar with diagnostics
from email_alert import is_email_configured, get_email_receivers, send_test_email

# Helper to mask email addresses for display
def _mask_email(addr: str) -> str:
    if not addr or "@" not in addr:
        return "(not set)"
    local, domain = addr.split("@", 1)
    masked_local = local[0] + "***" if len(local) > 1 else "*"
    return f"{masked_local}@{domain}"

env_file_exists = os.path.exists(".env")
if env_file_exists:
    st.sidebar.info("`.env` found in project root")
else:
    st.sidebar.info("`.env` not found in project root")

if st.sidebar.button("Reload .env"):
    # reload .env into current environment (use python-dotenv if available)
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except Exception:
        if os.path.exists('.env'):
            with open('.env', 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    os.environ[k] = v
    st.experimental_rerun()

if is_email_configured():
    receivers = get_email_receivers()
    sender = os.getenv('EMAIL_SENDER') or '(not set)'
    st.sidebar.success(f"Email configured — sender: {_mask_email(sender)}")
    st.sidebar.write(f"Recipients: {', '.join(receivers)}")
    if st.sidebar.button("Send test email"):
        ok = send_test_email()
        if ok:
            st.sidebar.success("✅ Test email sent — check recipient inbox")
        else:
            st.sidebar.error("❌ Test email failed — check configuration and logs")
else:
    st.sidebar.warning("Email not configured. Copy .env.example to .env and set EMAIL_SENDER, EMAIL_PASSWORD, and EMAIL_RECEIVERS.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = CSRNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ---------------- IMAGE UPLOAD ----------------
uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, channels="BGR", caption="Uploaded Image")

    # ✅ CORRECT preprocessing
    input_tensor = preprocess_frame(image)

    # Inference
    with torch.no_grad():
        density_map = model(input_tensor)
        crowd_count = int(density_map.sum().item())

    # ---------------- OUTPUT ----------------
    st.metric("👥 Predicted Crowd Count", crowd_count)

    if crowd_count > ALERT_THRESHOLD:
        st.error("🚨 Overcrowding Detected")
        if st.button("Send Email Alert"):
            success = send_email_alert(crowd_count)
            if success:
                st.success("✅ Email alert sent")
            else:
                st.error("❌ Failed to send email alert — check configuration and logs")
    else:
        st.success("✅ Crowd Level Normal")

    # ---------------- HEATMAP ----------------
    heatmap = generate_heatmap(image, density_map)
    # Convert BGR→RGB for Streamlit and display; use explicit width instead of deprecated use_column_width
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    st.image(heatmap_rgb, caption="Heatmap", width=image.shape[1])
