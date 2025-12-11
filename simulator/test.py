import cv2

for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"Camera index {i} OPENED")
        cap.release()
    else:
        print(f"Camera index {i} not available")