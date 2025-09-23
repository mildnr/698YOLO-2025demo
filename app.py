import sys, subprocess, importlib

def _pip_install(args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])

# ติดตั้งถ้าไม่มี
try:
    import ultralytics  # noqa
except ModuleNotFoundError:
    _pip_install(["-U", "pip"])
    _pip_install(["ultralytics"])
    _pip_install(["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"])
    _pip_install(["opencv-python-headless", "pillow", "numpy"])

from ultralytics import YOLO


import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("YOLO Image Detection App :)")

# Load YOLO model
# model = YOLO("runs/detect/train73/weights/best.pt")
model = YOLO("yolo11n.pt")

# Upload image
uploaded_image = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:

    # Show original image
    st.image(uploaded_image , caption="Uploaded Image", use_container_width=True)

    # Read image and convert to numpy array
    image = Image.open(uploaded_image)
    image_np = np.array(image)

    # Run YOLO inference
    st.info("Running YOLO object detection...")
    results = model.predict(image_np , conf=0.4)

    # Draw results on image
    result_image = results[0].plot()
    st.image(result_image , caption="YOLO Detection Result", use_container_width=True)
    st.success("Detection completed!")

    # Extract detection results
    boxes = results[0].boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)
    class_names = [model.names[i] for i in class_ids]

    # Count people
    person_count = class_names.count("person")
    st.write(f"Number of people detected: **{person_count}**")
