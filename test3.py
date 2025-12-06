import cv2

cam = cv2.VideoCapture(0)

while True:
    success, frame = cam.read()
    print(success)
