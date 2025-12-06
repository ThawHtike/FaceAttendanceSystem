from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
import pandas as pd
from datetime import datetime
import face_recognition
import numpy as np

app = Flask(__name__)

# ------------------ Setup ------------------

camera = cv2.VideoCapture(0)
os.makedirs("static/photos", exist_ok=True)


def init_files():
    if not os.path.exists("student_database.csv"):
        df = pd.DataFrame(columns=['student_id', 'student_name', 'photo'])
        df.to_csv("student_database.csv", index=False)

    if not os.path.exists("attendance.csv"):
        df = pd.DataFrame(columns=['student_id', 'student_name', 'date', 'time'])
        df.to_csv("attendance.csv", index=False)


init_files()

known_face_encodings = []
known_face_names = []
known_face_ids = []


# ------------------ The "No Restrictions" Logic ------------------

def universal_image_fix(img):
    """
    Accepts ANY image format (Gray, RGBA, 16-bit) and forces it
    into standard 3-channel 8-bit RGB that the AI can read.
    """
    # 1. If image is None (bad read), return None
    if img is None:
        return None

    # 2. Check current shape
    # Shape is usually (Height, Width, Channels)
    # Grayscale images might only have (Height, Width)

    # CASE A: Grayscale (2 Dimensions) -> Convert to 3-Channel RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # CASE B: 4-Channel (RGBA/Transparent) -> Convert to 3-Channel RGB
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    # CASE C: Standard BGR (OpenCV default) -> Convert to RGB
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. Force Data Type to uint8 (Standard 8-bit)
    # This fixes errors with high-bit depth images
    img = np.array(img, dtype=np.uint8)

    return img


def load_known_faces():
    global known_face_encodings, known_face_names, known_face_ids

    known_face_encodings = []
    known_face_names = []
    known_face_ids = []

    if not os.path.exists("student_database.csv"):
        return

    df = pd.read_csv("student_database.csv")

    for index, row in df.iterrows():
        img_path = f"static/photos/{row['photo']}"
        if os.path.exists(img_path):
            try:
                # Read using OpenCV (reads as BGR or Grayscale)
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                # Apply the Universal Fix
                rgb_img = universal_image_fix(img)

                if rgb_img is not None:
                    encodings = face_recognition.face_encodings(rgb_img)
                    if len(encodings) > 0:
                        known_face_encodings.append(encodings[0])
                        known_face_names.append(row['student_name'])
                        known_face_ids.append(str(row['student_id']))
                    else:
                        print(f"No face found in {img_path}")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")


load_known_faces()


# ------------------ Attendance Logic ------------------

def mark_attendance(student_id, student_name):
    now = datetime.now()
    today_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    try:
        df = pd.read_csv("attendance.csv")
    except:
        df = pd.DataFrame(columns=['student_id', 'student_name', 'date', 'time'])

    already_present = df[(df['student_id'].astype(str) == str(student_id)) & (df['date'] == today_date)]

    if already_present.empty:
        new_row = pd.DataFrame({
            'student_id': [student_id],
            'student_name': [student_name],
            'date': [today_date],
            'time': [current_time]
        })
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv("attendance.csv", index=False)
        print(f"Marked: {student_name}")


def gen_frames():
    global camera
    while True:
        success, frame = camera.read()
        if not success:
            continue

        try:
            # Resize
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Apply Universal Fix (Ensures video feed never crashes)
            # Note: Camera is usually BGR, so we just need BGR->RGB conversion
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            rgb_small_frame = np.ascontiguousarray(rgb_small_frame, dtype=np.uint8)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"

                if len(known_face_encodings) > 0:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        stu_id = known_face_ids[best_match_index]
                        mark_attendance(stu_id, name)

                top, right, bottom, left = face_location
                top *= 4;
                right *= 4;
                bottom *= 4;
                left *= 4

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception:
            pass


# ------------------ Routes ------------------

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        student_id = request.form['student_id']
        student_name = request.form['student_name']
        photo = request.files['photo']

        if photo:
            filename = f"{student_id}_{photo.filename.replace(' ', '_')}"
            save_path = f"static/photos/{filename}"

            # 1. Read Raw Bytes
            file_bytes = np.frombuffer(photo.read(), np.uint8)
            # 2. Decode UNCHANGED (Keep alpha channel, keep grayscale as-is)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

            if img is not None:
                # 3. Apply Universal Fix (Convert to RGB for DB logic)
                # We need to save a clean version to disk to avoid future errors
                rgb_clean = universal_image_fix(img)

                # Convert back to BGR for saving with OpenCV
                bgr_save = cv2.cvtColor(rgb_clean, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, bgr_save)

                # Database Entry
                df = pd.read_csv("student_database.csv")
                new_row = pd.DataFrame(
                    {'student_id': [student_id], 'student_name': [student_name], 'photo': [filename]})
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv("student_database.csv", index=False)

                load_known_faces()
                return redirect("/students")

    return render_template("register.html")


@app.route('/students')
def students():
    if os.path.exists("student_database.csv"):
        df = pd.read_csv("student_database.csv")
        return render_template("students.html", tables=df.to_dict('records'))
    return "No database found."


@app.route('/attendance')
def attendance_page():
    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv")
        if not df.empty:
            df = df.sort_values(by=['date', 'time'], ascending=False)
        return render_template("attendance.html", tables=df.to_dict('records'))
    return render_template("attendance.html", tables=[])


@app.route('/live_feed_page')
def live_feed_page():
    return render_template("live_attendance.html")


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        camera.release()