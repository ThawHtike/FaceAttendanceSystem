FaceAttendanceSystem - README
-----------------------------

1) Setup (recommended):
   python -m venv .venv
   .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt

2) Run:
   python app.py
   Open http://127.0.0.1:5000

3) Notes & troubleshooting:
   - If you get "PermissionError" when registering:
     Close Excel or any program that has opened student_database.csv.
   - If uploaded images fail:
     Make sure you upload JPG/PNG (not HEIC). The app attempts to normalize images automatically,
     but HEIC sometimes fails — convert on phone to JPG first, or take screenshots that are JPG.
   - face_recognition requires dlib. On Windows you may need to install Visual Studio Build Tools
     or use prebuilt wheels for dlib.
   - If live camera not showing: allow camera access; try changing cv2.VideoCapture(0) to (1) or use CAP_DSHOW:
     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

4) Files:
   - student_database.csv : student_id, student_name, image_filename
   - attendance.csv       : student_id, student_name, time, date

