import streamlit as st
import cv2
import pandas as pd
import os
import numpy as np
import base64
from datetime import datetime, timedelta
from PIL import Image, ImageOps
import face_recognition

# =====================================================
# CONFIG & DIRECTORIES
# =====================================================
PHOTO_DIR = "static/photos"
DB_FILE = "student_database.csv"
ATT_FILE = "attendance.csv"
CLASS_DURATION_HOURS = 3

os.makedirs(PHOTO_DIR, exist_ok=True)


# =====================================================
# INITIALIZE DATABASE FILES
# =====================================================
def init_files():
    if not os.path.exists(DB_FILE):
        pd.DataFrame(columns=["student_id", "student_name", "photo"]).to_csv(DB_FILE, index=False)
    if not os.path.exists(ATT_FILE):
        pd.DataFrame(columns=["student_id", "student_name", "date", "time"]).to_csv(ATT_FILE, index=False)


init_files()


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def get_base64_image(image_path):
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                return f"data:image/jpeg;base64,{base64.b64encode(img_file.read()).decode()}"
        return None
    except Exception:
        return None


def load_image(file_or_path):
    img = Image.open(file_or_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    return np.array(img)


def get_known_faces():
    if not os.path.exists(DB_FILE): return [], []
    df_db = pd.read_csv(DB_FILE)
    known_encodings, known_metadata = [], []
    for _, row in df_db.iterrows():
        path = os.path.join(PHOTO_DIR, row["photo"])
        if os.path.exists(path):
            img = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(img)
            if encs:
                known_encodings.append(encs[0])
                known_metadata.append({"id": row["student_id"], "name": row["student_name"]})
    return known_encodings, known_metadata


def already_marked_this_class(student_id):
    df = pd.read_csv(ATT_FILE)
    # Ensure ID is compared as string
    records = df[df["student_id"].astype(str) == str(student_id)]
    if records.empty: return False

    # Check the last record for this student
    last = records.iloc[-1]
    last_time = datetime.strptime(f"{last['date']} {last['time']}", "%Y-%m-%d %H:%M:%S")

    # If time elapsed is less than 3 hours, return True (already marked)
    return (datetime.now() - last_time) < timedelta(hours=CLASS_DURATION_HOURS)


def save_attendance(student_id, student_name):
    now = datetime.now()
    new_entry = pd.DataFrame([{"student_id": student_id, "student_name": student_name,
                               "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S")}])
    new_entry.to_csv(ATT_FILE, mode="a", header=False, index=False)


# =====================================================
# UI - PAGE CONFIG & STYLING
# =====================================================
st.set_page_config(page_title="Face Attendance System", page_icon="📷", layout="wide")

st.markdown("""
    <style>
    [data-testid="stDataFrame"] { margin: 0 auto; display: flex; justify-content: center; }
    th { text-align: center !important; }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width="stretch")

st.sidebar.title("Beykoz University")
menu = ["Register Student", "Live Attendance", "View Records"]
# 2026 FIX: Added non-empty label "Navigation"
choice = st.sidebar.selectbox("Navigation", menu, label_visibility="collapsed")
st.sidebar.write("---")

if os.path.exists(ATT_FILE):
    df_download = pd.read_csv(ATT_FILE)
    st.sidebar.download_button(
        label="📥 Download Attendance Record",
        data=df_download.to_csv(index=False).encode('utf-8'),
        file_name=f"Attendance_Report_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        width="stretch"
    )

st.sidebar.markdown(f"""
<div style="font-size:12px; line-height:1.5; color: #555; margin-top: 20px;">
    <strong>MASTER’S TERM PROJECT</strong><br>
    <strong>COMPUTER ENGINEERING DEPARTMENT</strong><br><br>
    <strong>Student:</strong> Thaw Zin Htike<br>
    <strong>ID:</strong> 2430210021<br><br>
    <strong>MASTER’S PROJECT ADVISOR</strong><br>
    <strong>Prof. Oruç Raif Önvural</strong>
</div>
""", unsafe_allow_html=True)

# =====================================================
# MAIN CONTENT
# =====================================================

if choice == "Register Student":
    st.header("👤 Student Registration")
    with st.form("reg_form", clear_on_submit=True):
        sid = st.text_input("Student ID (Unique)")
        name = st.text_input("Full Name")
        file = st.file_uploader("Upload Clear Face Photo", type=["jpg", "jpeg", "png"])
        submit = st.form_submit_button("Register Student")

    if submit:
        if not sid or not name or not file:
            st.error("Please fill all fields and upload a photo.")
        else:
            # 1. Check Duplicate ID
            df_db = pd.read_csv(DB_FILE)
            if str(sid) in df_db["student_id"].astype(str).values:
                st.error(f"❌ Error: Student ID '{sid}' is already registered!")
            else:
                img = load_image(file)
                encodings = face_recognition.face_encodings(img)

                if not encodings:
                    st.error("❌ No face detected. Please use a clearer photo.")
                else:
                    new_encoding = encodings[0]
                    # 2. Check Duplicate Face
                    known_encs, known_meta = get_known_faces()
                    is_duplicate_face = False

                    if known_encs:
                        matches = face_recognition.compare_faces(known_encs, new_encoding, tolerance=0.5)
                        if True in matches:
                            idx = matches.index(True)
                            st.error(
                                f"❌ Registration Failed: This face is already registered as **{known_meta[idx]['name']}**.")
                            is_duplicate_face = True

                    if not is_duplicate_face:
                        filename = f"{sid}_{name.replace(' ', '_')}.jpg"
                        path = os.path.join(PHOTO_DIR, filename)
                        Image.fromarray(img).save(path)

                        new_student = pd.DataFrame([[sid, name, filename]],
                                                   columns=["student_id", "student_name", "photo"])
                        new_student.to_csv(DB_FILE, mode='a', header=False, index=False)
                        st.success(f"✅ Registered {name} successfully!")
                        st.image(img, width=200)

elif choice == "Live Attendance":
    st.header("🎥 Live Recognition")
    known_encodings, known_metadata = get_known_faces()

    if not known_encodings:
        st.warning("No students registered yet.")
    else:
        run_cam = st.checkbox("Start Camera")
        status_placeholder = st.empty()
        frame_placeholder = st.empty()

        if run_cam:
            video_capture = cv2.VideoCapture(0)
            # Use session state to avoid spamming the UI during the same camera run
            session_marked = set()

            while run_cam:
                ret, frame = video_capture.read()
                if not ret: break

                # Resize for performance
                rgb_small = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=0.25, fy=0.25), cv2.COLOR_BGR2RGB)
                face_locs = face_recognition.face_locations(rgb_small)
                face_encs = face_recognition.face_encodings(rgb_small, face_locs)

                for face_encoding in face_encs:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)

                    if True in matches:
                        idx = matches.index(True)
                        sid = known_metadata[idx]["id"]
                        name = known_metadata[idx]["name"]

                        if sid not in session_marked:
                            if not already_marked_this_class(sid):
                                save_attendance(sid, name)
                                session_marked.add(sid)
                                status_placeholder.markdown(
                                    f"<h3 style='color: #28a745; text-align: center;'>✅ Attendance Marked: {name}</h3>",
                                    unsafe_allow_html=True)
                            else:
                                session_marked.add(sid)
                                # UPDATED ERROR MESSAGE
                                status_placeholder.markdown(
                                    f"<h3 style='color: #ff9800; text-align: center;'>⚠️ {name} already took attendance for this class.</h3>",
                                    unsafe_allow_html=True)

                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")
            video_capture.release()

elif choice == "View Records":
    st.header("📊 Data Records")
    tab1, tab2 = st.tabs(["Attendance Logs", "Student Database"])

    with tab1:
        if os.path.exists(ATT_FILE):
            att_df = pd.read_csv(ATT_FILE).sort_values(by=["date", "time"], ascending=False)
            if not att_df.empty:
                att_df.insert(0, "S.No", range(1, len(att_df) + 1))
                st.dataframe(att_df, width="stretch", hide_index=True)
            else:
                st.info("No records found.")

    with tab2:
        if os.path.exists(DB_FILE):
            db_df = pd.read_csv(DB_FILE)
            if not db_df.empty:
                db_df.insert(0, "S.No", range(1, len(db_df) + 1))
                db_df["photo"] = db_df["photo"].apply(lambda x: get_base64_image(os.path.join(PHOTO_DIR, x)))
                st.dataframe(
                    db_df,
                    column_config={"photo": st.column_config.ImageColumn("Student Photo", width="medium")},
                    hide_index=True, width="stretch"
                )