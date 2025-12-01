import os
import time
import cv2
import joblib
import pandas as pd
import numpy as np
from datetime import date, datetime
from flask import Flask, render_template, request
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

MESSAGE = "WELCOME - Instruction: To register your attendance kindly click on 'a' on the keyboard"
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ensure directories exist
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)
if not os.path.exists(f'Attendance/Attendance-{datetoday}.csv'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    if img is None or img.size == 0:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_detector.detectMultiScale(gray, 1.3, 5)

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            if img is not None:
                resized_face = cv2.resize(img, (50, 50))
                faces.append(resized_face.ravel())
                labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    return df['Name'], df['Roll'], df['Time'], len(df)

def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if str(userid) not in df['Roll'].astype(str).values:
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')
    else:
        print("User already marked attendance. Skipping duplicate.")

@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times,
                           l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=MESSAGE)

@app.route('/start', methods=['GET'])
def start():
    ATTENDANCE_MARKED = False

    if not os.path.exists('static/face_recognition_model.pkl'):
        names, rolls, times, l = extract_attendance()
        msg = 'Face not registered. Please add yourself first.'
        return render_template('home.html', names=names, rolls=rolls, times=times,
                               l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=msg)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Camera not accessible."

    time.sleep(2)  # Let camera warm up

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            cv2.putText(frame, f'{identified_person}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if cv2.waitKey(1) == ord('a'):
                add_attendance(identified_person)
                ATTENDANCE_MARKED = True
                break

        cv2.imshow('Attendance - Press "a" to mark or "q" to quit', frame)

        if ATTENDANCE_MARKED or cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times,
                           l=l, totalreg=totalreg(), datetoday2=datetoday2, mess="Attendance taken successfully.")

@app.route('/add', methods=['POST'])
def add_user():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'

    os.makedirs(userimagefolder, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Camera not accessible."

    time.sleep(2)  # Let camera warm up

    i, j = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)

            if j % 10 == 0:
                face_img = frame[y:y+h, x:x+w]
                cv2.imwrite(f'{userimagefolder}/{newusername}_{i}.jpg', face_img)
                i += 1
            j += 1

        cv2.imshow('Registering Face - Press ESC to stop', frame)
        if j >= 500 or i >= 50 or cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    train_model()

    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times,
                           l=l, totalreg=totalreg(), datetoday2=datetoday2,
                           mess="User added and model trained.")

if __name__ == '__main__':
    app.run(debug=True, port=8080)

