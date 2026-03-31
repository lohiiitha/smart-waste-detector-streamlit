import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile

st.title("Smart Waste Detector")

@st.cache_resource
def load_model():
    return YOLO("weights/best.pt")

model = load_model()

# IMAGE
st.header("Image Detection")
uploaded_image = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image)

    img_np = np.array(image)
    results = model(img_np)

    result_img = results[0].plot()
    st.image(result_img)

# VIDEO
st.header("Video Detection")
uploaded_video = st.file_uploader("Upload Video", type=["mp4"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()
        stframe.image(annotated, channels="BGR")

    cap.release()