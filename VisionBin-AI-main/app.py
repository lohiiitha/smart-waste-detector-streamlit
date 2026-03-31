import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(layout="wide")
st.title("Smart Waste Detector")

# Load model (cache to avoid reload)
@st.cache_resource
def load_model():
    return YOLO("weights/best.pt")

model = load_model()

# ---------------- IMAGE DETECTION ----------------
st.header("📷 Image Detection")

uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_np = np.array(image)
    results = model(img_np, conf=0.4)

    detected = list(set([model.names[int(b.cls[0])] for b in results[0].boxes]))
    status = "Detected: " + ", ".join(detected) if detected else "None Detected"

    result_img = results[0].plot()
    st.image(result_img, caption=status, use_column_width=True)

# ---------------- VIDEO DETECTION ----------------
st.header("🎥 Video Detection")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    status_text = st.empty()

    cumulative_found = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.4)
        found = set([model.names[int(b.cls[0])] for b in results[0].boxes])
        cumulative_found.update(found)

        status = "Detected: " + ", ".join(found) if found else "Scanning..."
        status_text.text(status)

        annotated = results[0].plot()
        stframe.image(annotated, channels="BGR")

    cap.release()

    if cumulative_found:
        status_text.text("Finished! Found: " + ", ".join(cumulative_found))
    else:
        status_text.text("Finished! Found: Nothing")

# ---------------- WEBCAM (LIMITED SUPPORT) ----------------
st.header("📹 Webcam (Experimental)")

run_cam = st.button("Start Camera")

if run_cam:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.4)
        annotated = results[0].plot()

        stframe.image(annotated, channels="BGR")

    cap.release()