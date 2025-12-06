from PIL import Image
import os

path = r"C:\Users\thawz\PycharmProjects\FaceAttendanceSystem\IMAGE_FILES"   # change to your folder

for file in os.listdir(path):
    try:
        img = Image.open(os.path.join(path, file))
        print(file, img.mode)
    except Exception as e:
        print("CORRUPTED:", file, e)
