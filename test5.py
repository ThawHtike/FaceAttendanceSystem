import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# -----------------------------
# MediaPipe Face Detector
# -----------------------------
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

# -----------------------------
# Face Detection Function
# -----------------------------
def detect_and_crop_face(img):
    """
    img: BGR image (OpenCV format)
    return: cropped face or None
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)

    if not results.detections:
        return None

    h, w, _ = img.shape
    box = results.detections[0].location_data.relative_bounding_box

    x1 = int(box.xmin * w)
    y1 = int(box.ymin * h)
    x2 = int((box.xmin + box.width) * w)
    y2 = int((box.ymin + box.height) * h)

    # Safety clamp
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    face = img[y1:y2, x1:x2]

    if face.size == 0:
        return None

    return face

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Face Detection App", layout="centered")

st.title("🧑 Face Detection (MediaPipe)")
st.write("Upload an image and the app will detect and crop the face.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # Convert RGB -> BGR for OpenCV
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    # Detect face
    face = detect_and_crop_face(img_bgr)

    if face is None:
        st.error("❌ No face detected.")
    else:
        st.success("✅ Face detected!")
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        st.subheader("Detected Face")
        st.image(face_rgb, width=250)
