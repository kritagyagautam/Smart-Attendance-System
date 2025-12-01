import cv2

print("Checking available cameras...")
for i in range(5):  # check first 5 device IDs (0-4)
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available")
        cap.release()
    else:
        print(f"Camera {i} is NOT available")
