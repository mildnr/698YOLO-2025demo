import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

st.title("YOLO License Plate Detection")

# Load YOLO model (โมเดลป้ายทะเบียนของเรา)
# model = YOLO("runs/detect/train73/weights/best.pt")
model = YOLO("best.pt")

# ช่วยหา class id ของ "ป้ายทะเบียน" จาก model.names
def get_plate_class_ids(names: dict):
    # รองรับหลายชื่อ: license-plate / license_plate / plate / lp
    synonyms = {"license-plate", "license_plate", "licence-plate", "licence_plate", "plate", "lp"}
    ids = []
    for cid, cname in names.items():
        key = cname.lower().replace(" ", "").replace("-", "_")
        if key in synonyms:
            ids.append(cid)
    # กรณีโมเดลมีแค่คลาสเดียว (ป้ายทะเบียนอย่างเดียว)
    if not ids and len(names) == 1:
        ids = [0]
    return ids

# Upload image
uploaded_image = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:

    # Show original image
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    # Read image and convert to numpy array (แก้ EXIF เอียงด้วย)
    image = Image.open(uploaded_image).convert("RGB")
    image = ImageOps.exif_transpose(image)
    image_np = np.array(image)

    # ระบุ class ids ของป้ายทะเบียน
    plate_ids = get_plate_class_ids(model.names)

    # Run YOLO inference (ปรับค่าตามที่เทสแล้วเหมาะกับงานป้ายทะเบียน)
    st.info("Running YOLO license plate detection...")
    results = model.predict(
        image_np,
        conf=0.30,      # จากกราฟ F1–Confidence ที่ได้
        iou=0.60,
        imgsz=896,
        classes=plate_ids if plate_ids else None,  # ถ้าระบุได้ให้กรองเฉพาะป้ายทะเบียน
        verbose=False
    )

    # Draw results on image
    result_image = results[0].plot()
    st.image(result_image, caption="YOLO Detection Result", use_container_width=True)
    st.success("Detection completed!")

    # Extract detection results
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        st.warning("ไม่พบป้ายทะเบียนตาม threshold ที่ตั้งไว้ (ลองลด conf ลงเล็กน้อย)")
    else:
        class_ids = boxes.cls.cpu().numpy().astype(int)
        class_names = [model.names[i] for i in class_ids]
        confs = boxes.conf.cpu().numpy()

        # ถ้าไม่ได้กรอง classes ตอน predict (เช่นหา class ไม่เจอ) ค่อยกรองซ้ำที่นี่
        synonyms = {"license-plate", "license_plate", "licence-plate", "licence_plate", "plate", "lp"}
        if plate_ids:
            plate_count = len(class_ids)
            keep_idx = list(range(len(class_ids)))
        else:
            keep_idx = [k for k, n in enumerate(class_names)
                        if n.lower().replace(" ", "").replace("-", "_") in synonyms]
            plate_count = len(keep_idx)

        st.write(f"จำนวนป้ายทะเบียนที่ตรวจพบ: **{plate_count}**")

        # (เสริมเล็กน้อย) โชว์รายละเอียดกล่องที่เจอ
        if plate_count > 0:
            with st.expander("รายละเอียดผลตรวจจับ"):
                for j, k in enumerate(keep_idx, start=1):
                    st.write(f"- #{j}: class = **{class_names[k]}**, conf = **{confs[k]:.3f}**")
