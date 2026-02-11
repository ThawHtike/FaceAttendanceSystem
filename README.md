This is a professional `README.md` file tailored for a Master’s Term Project. It includes all the technical details of the system we built, the installation steps, and the academic information required for your submission.

---

 Face-Recognition Based Attendance System

Master’s Term Project Computer Engineering Department, Beykoz University

 🎓 Academic Information

 Student: Thaw Zin Htike
 Student ID: 2430210021
 Project Advisor: Prof. Oruç Raif Önvural
 Academic Year: 2025-2026

---

 📝 Project Overview

This project is an automated attendance management system using Biometric Facial Recognition. It is designed to replace traditional manual attendance methods with a faster, more secure, and paperless digital solution. The system is built using Python, Streamlit for the web interface, and the `face_recognition` library (dlib-based) for high-accuracy face matching.

 ✨ Key Features

 Secure Registration: Prevents duplicate entries by checking both Student ID (database check) and Facial Features (biometric check).
 Live Recognition: Real-time face detection and identification via webcam.
 Session Control: Implements a 3-hour lockout logic; a student cannot mark attendance twice for the same class session.
 Data Management: View registration records and attendance logs with timestamps and student photos.
 Export Functionality: Sidebar option to download the attendance report as a `.csv` file for administrative use.
 Modern UI: Responsive dashboard compliant with Streamlit 2026 standards.

---

 🛠️ Tech Stack

 Frontend: Streamlit
 Computer Vision: OpenCV (`cv2`)
 Biometrics: `face_recognition` (utilizing HOG and Linear SVM)
 Data Handling: Pandas & NumPy
 Image Processing: Pillow (PIL)

---

 🚀 Getting Started

 1. Prerequisites

Ensure you have Python 3.9+ installed. You will also need `cmake` to install the face recognition library on some systems.

 2. Installation

Clone this repository or extract the project folder, then install the required libraries:

```bash
pip install streamlit opencv-python pandas numpy face_recognition Pillow

```

 3. Folder Structure

Ensure your project directory looks like this:

```text
FaceAttendanceSystem/
├── app.py                   Main application code
├── logo.png                 University logo (optional)
├── static/
│   └── photos/              Stores registered student images
├── student_database.csv     Stores student IDs and names
└── attendance.csv           Stores attendance logs

```

 4. Running the App

Run the following command in your terminal:

```bash
streamlit run app.py

```

---

 📊 System Logic

1. Registration: The system extracts a 128-dimensional face encoding from the uploaded photo. It compares this encoding against the entire database to ensure the person isn't already registered under a different ID.
2. Recognition: During live stream, the system resizes frames to  size for faster processing. It matches detected faces against the `known_encodings`.
3. Attendance Rules:  If a face is recognized and has not been marked in the last 3 hours, a record is saved.
 If recognized again within 3 hours, the system displays: "Already took attendance for this class."



---

 📜 License

This project was developed as a Master’s Term Project for academic purposes at Beykoz University.
