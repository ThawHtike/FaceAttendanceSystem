import streamlit as st
import cv2
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
from PIL import Image, ImageOps
import face_recognition

# =====================================================
# CONFIG
# =====================================================
PHOTO_DIR = "static/photos"
DB_FILE = "student_database.csv"
ATT_FILE = "attendance.csv"
CLASS_DURATION_HOURS = 3

os.makedirs(PHOTO_DIR, exist_ok=True)

# =====================================================
# INIT FILES
# =====================================================
if not os.path.exists(DB_FILE):
    pd.DataFrame(
        columns=["student_id", "student_name", "photo"]
    ).to_csv(DB_FILE, index=False)

if not os.path.exists(ATT_FILE):
    pd.DataFrame(
        columns=["student_id", "student_name", "date", "time"]
    ).to_csv(ATT_FILE, index=False)

# =====================================================
# IMAGE LOADER
# =====================================================
def load_image(file_or_path):
    if isinstance(file_or_path, str):
        img = Image.open(file_or_path)
    else:
        file_or_path.seek(0)
        img = Image.open(file_or_path)

    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    return np.array(img)

# =====================================================
# CHECK 3-HOUR RULE
# =====================================================
def already_marked_this_class(student_id):
    df = pd.read_csv(ATT_FILE)
    records = df[df["student_id"].astype(str) == str(student_id)]

    if records.empty:
        return False

    last = records.iloc[-1]
    last_time = datetime.strptime(
        f"{last['date']} {last['time']}",
        "%Y-%m-%d %H:%M:%S"
    )

    return datetime.now() - last_time < timedelta(hours=CLASS_DURATION_HOURS)

# =====================================================
# SAVE ATTENDANCE
# =====================================================
def save_attendance(student_id, student_name):
    now = datetime.now()
    pd.DataFrame([{
        "student_id": student_id,
        "student_name": student_name,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S")
    }]).to_csv(ATT_FILE, mode="a", header=False, index=False)

# =====================================================
# UI
# =====================================================
st.set_page_config(page_title="Face Attendance", page_icon="📷")
st.title("📷 Face Attendance System (Photo Matching)")

menu = ["Register Student", "Live Attendance", "View Records"]
choice = st.sidebar.selectbox("Menu", menu)

# =====================================================
# REGISTER STUDENT
# =====================================================
if choice == "Register Student":
    st.header("Register Student")

    with st.form("register"):
        sid = st.text_input("Student ID")
        name = st.text_input("Student Name")
        file = st.file_uploader("Upload Face Photo", type=["jpg", "jpeg", "png"])
        submit = st.form_submit_button("Register")

    if submit:
        if not sid or not name or not file:
            st.warning("Fill all fields")
        else:
            img = load_image(file)
            encodings = face_recognition.face_encodings(img)

            if not encodings:
                st.error("❌ No face detected")
            else:
                df = pd.read_csv(DB_FILE)

                if sid in df["student_id"].astype(str).values:
                    st.error("Student already exists")
                else:
                    filename = f"{sid}_{name}.jpg"
                    path = os.path.join(PHOTO_DIR, filename)
                    Image.fromarray(img).save(path)

                    df.loc[len(df)] = [sid, name, filename]
                    df.to_csv(DB_FILE, index=False)

                    st.success("✅ Student Registered")
                    st.image(img)

# =====================================================
# LIVE ATTENDANCE (MULTI STUDENT, PHOTO MATCH)
# =====================================================
elif choice == "Live Attendance":
    st.header("Live Attendance (One record per student / 3 hours)")

    df_db = pd.read_csv(DB_FILE)
    if df_db.empty:
        st.warning("No registered students")
    else:
        if "camera_on" not in st.session_state:
            st.session_state.camera_on = False

        if st.button("▶ Start Camera"):
            st.session_state.camera_on = True

        if st.button("⏹ Stop Camera"):
            st.session_state.camera_on = False

        frame_box = st.image([])
        status = st.empty()

        # Preload known faces
        known_encodings = []
        known_ids = []
        known_names = []

        for _, row in df_db.iterrows():
            img = load_image(os.path.join(PHOTO_DIR, row["photo"]))
            enc = face_recognition.face_encodings(img)
            if enc:
                known_encodings.append(enc[0])
                known_ids.append(row["student_id"])
                known_names.append(row["student_name"])

        if st.session_state.camera_on:
            cam = cv2.VideoCapture(0)

            while st.session_state.camera_on:
                ret, frame = cam.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb)
                face_encs = face_recognition.face_encodings(rgb, face_locations)

                for enc, loc in zip(face_encs, face_locations):
                    matches = face_recognition.compare_faces(
                        known_encodings, enc, tolerance=0.45
                    )

                    if True in matches:
                        idx = matches.index(True)
                        sid = known_ids[idx]
                        name = known_names[idx]

                        if already_marked_this_class(sid):
                            status.info(f"ℹ {name} already marked for this class")
                        else:
                            save_attendance(sid, name)
                            status.success(f"✅ Attendance saved: {name}")

                        top, right, bottom, left = loc
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(
                            frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
                        )

                frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            cam.release()

# =====================================================
# VIEW RECORDS
# =====================================================
elif choice == "View Records":
    st.header("Attendance Records")
    st.dataframe(pd.read_csv(ATT_FILE))
