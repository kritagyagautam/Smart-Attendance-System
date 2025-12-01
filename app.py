# import time
# from flask import session
# import cv2
# import os, cv2, joblib, pandas as pd, numpy as np
# from datetime import date, datetime
# from flask import Flask, render_template, request, Response, flash, redirect, url_for
# from sklearn.neighbors import KNeighborsClassifier


# app = Flask(__name__)
# app.secret_key = "your_secret_key_here"  
# MESSAGE = "WELCOME - Instruction: To register your attendance kindly click on 'a' on the keyboard"
# datetoday = date.today().strftime("%m_%d_%y")
# datetoday2 = date.today().strftime("%d-%B-%Y")
# face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# capture_user = None  # Stores (username, userid)
# capture_count = 0
# capture_limit = 50
# user_folder = None


# # Ensure directories exist
# os.makedirs('Attendance', exist_ok=True)
# os.makedirs('static/faces', exist_ok=True)
# if not os.path.exists(f'Attendance/Attendance-{datetoday}.csv'):
#     with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
#         f.write('Name,Roll,Time')


# def totalreg():
#     return len(os.listdir('static/faces'))

# def extract_faces(img):
#     if img is None or img.size == 0:
#         return []
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return face_detector.detectMultiScale(gray, 1.3, 5)

# def identify_face(facearray):
#     model = joblib.load('static/face_recognition_model.pkl')
#     return model.predict(facearray)

# def train_model():
#     faces = []
#     labels = []
#     userlist = os.listdir('static/faces')
#     for user in userlist:
#         for imgname in os.listdir(f'static/faces/{user}'):
#             img = cv2.imread(f'static/faces/{user}/{imgname}')
#             if img is not None:
#                 resized_face = cv2.resize(img, (50, 50))
#                 faces.append(resized_face.ravel())
#                 labels.append(user)
#     faces = np.array(faces)
#     knn = KNeighborsClassifier(n_neighbors=5)
#     knn.fit(faces, labels)
#     joblib.dump(knn, 'static/face_recognition_model.pkl')

# def extract_attendance():
#     df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
#     return df['Name'], df['Roll'], df['Time'], len(df)

# def add_attendance(name):
#     # Split only on the last underscore
#     if '_' not in name:
#         print(f"Invalid name format: {name}")
#         return
#     username, userid = name.rsplit('_', 1)
#     current_time = datetime.now().strftime("%H:%M:%S")
#     df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
#     if str(userid) not in df['Roll'].astype(str).values:
#         with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
#             f.write(f'\n{username},{userid},{current_time}')
#     else:
#         print("User already marked attendance. Skipping duplicate.")

# @app.route('/add-admin', methods=['POST'])
# def add_admin():
#     username = request.form['new_admin_username']
#     password = request.form['new_admin_password']

#     with open("admin_users.csv", "a") as f:
#         f.write(f"{username},{password}\n")

#     return redirect('/admin-dashboard')

# @app.route('/')
# def home():
#     names, rolls, times, l = extract_attendance()
#     return render_template('home.html', names=names, rolls=rolls, times=times,
#                            l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=MESSAGE)



# @app.route('/admin-login', methods=['GET', 'POST'])
# def admin_login():
#     if request.method == 'POST':
#         username = request.form.get('username')
#         password = request.form.get('password')

#         valid = False

#         with open('admin_users.csv', 'r') as f:
#             reader = csv.reader(f)
#             for row in reader:
#                 if len(row) < 2:
#                     continue
#                 if username == row[0] and password == row[1]:
#                     valid = True
#                     break

#         if valid:
#             session['admin'] = username
#             flash('Login successful!', 'success')
#             return redirect('/admin-dashboard')
#         else:
#             flash('Invalid credentials!', 'danger')
#             return redirect('/admin-login')

#     return render_template('adminlogin.html')



# @app.route('/admin-logout')
# def admin_logout():
#     session.pop('admin', None)
#     flash('Logged out successfully!', 'info')
#     return redirect('/admin-login')



# ####
# @app.route('/dashboard')
# def dashboard():
#     names, rolls, times, l = extract_attendance()
#     attendance_list = [(i+1, names[i], rolls[i], times[i]) for i in range(l)]
#     return render_template('dashboard.html', attendance_list=attendance_list, l=l)




# # # TODO: Add database signup logic here
# # flash(f'Account created for {new_user}!', 'success')

# # return render_template('adminlogin.html')

# ###########
# import csv
# from flask import request, render_template, redirect, url_for, flash

# @app.route('/admin-signup', methods=['GET', 'POST'])
# def admin_signup():
#     if request.method == 'POST':
#         username = request.form.get('username')
#         password = request.form.get('password')

#         with open('admin_users.csv', 'r') as f:
#             reader = csv.reader(f)
#             for row in reader:
#                 if len(row) >= 2 and row[0] == username:
#                     flash('Username already exists!', 'danger')
#                     return redirect('/admin-signup')

#         with open('admin_users.csv', 'a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow([username, password])

#         flash('Account created successfully!', 'success')
#         return redirect('/admin-login')

#     return render_template('adminsignup.html')



# ##########
# @app.route('/admin-dashboard')
# def admin_dashboard():
#     return render_template('dashboard.html')

# # Global variable to store last identified person
# identified_person_global = None

# def gen_frames():
#     global identified_person_global
#     cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Use MacBook camera
#     time.sleep(2)  # camera warmup
#     if not cap.isOpened():
#         return  # could add logging

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         faces = extract_faces(frame)

#         for (x, y, w, h) in faces:
#             face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
#             identified_person_global = identify_face(face.reshape(1, -1))[0]
#             cv2.putText(frame, f'{identified_person_global}', (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#     cap.release()

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



# @app.route('/start', methods=['GET', 'POST'])
# def start():
#     global identified_person_global

#     if not os.path.exists('static/face_recognition_model.pkl'):
#         names, rolls, times, l = extract_attendance()
#         msg = 'Face not registered. Please add yourself first.'
#         return render_template('home.html', names=names, rolls=rolls, times=times,
#                                l=l, totalreg=totalreg(), datetoday2=datetoday2, mess=msg)

#     if request.method == 'POST':
#         if identified_person_global:
#             add_attendance(identified_person_global)
#             msg = f"Attendance marked for {identified_person_global}"
#         else:
#             msg = "No face detected yet."

#         names, rolls, times, l = extract_attendance()
#         # Rebuild attendance_list
#         attendance_list = [(i+1, names[i], rolls[i], times[i]) for i in range(l)]
        
#         return render_template('home.html',
#                             attendance_list=attendance_list,
#                             l=l,
#                             totalreg=totalreg(),
#                             datetoday2=datetoday2,
#                             mess=msg)


#     # GET request: show live camera page
#     names, rolls, times, l = extract_attendance()
#     attendance_list = [(i+1, names[i], rolls[i], times[i]) for i in range(l)]
#     return render_template('home.html',
#                            attendance_list=attendance_list,
#                            l=l,
#                            totalreg=totalreg(),
#                            datetoday2=datetoday2)


# def gen_registration_frames():
#     global capture_user, capture_count, user_folder
#     cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # MacBook camera
#     time.sleep(2)
#     if not cap.isOpened():
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         faces = extract_faces(frame)
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
#             cv2.putText(frame, f'Images Captured: {capture_count}/{capture_limit}', (30, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#     cap.release()

# @app.route('/registration_video_feed')
# def registration_video_feed():
#     return Response(gen_registration_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/add', methods=['GET', 'POST'])
# def add_user():
#     global capture_user, capture_count, capture_limit, user_folder

#     if request.method == 'POST':
#         # Initialize user for capture
#         newusername = request.form['newusername']
#         newuserid = request.form['newuserid']
#         capture_user = (newusername, newuserid)
#         user_folder = f'static/faces/{newusername}_{newuserid}'
#         os.makedirs(user_folder, exist_ok=True)
#         capture_count = 0
#         capture_limit = 50

#         # Start automated capture
#         cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
#         time.sleep(1)
#         while capture_count < capture_limit:
#             ret, frame = cap.read()
#             if not ret:
#                 continue

#             faces = extract_faces(frame)
#             for (x, y, w, h) in faces:
#                 face_img = frame[y:y+h, x:x+w]
#                 cv2.imwrite(f'{user_folder}/{newusername}_{capture_count}.jpg', face_img)
#                 capture_count += 1
#                 break  # one face per frame

#         cap.release()
#         train_model()

#         # Automatically add first attendance entry for today
#         add_attendance(f"{newusername}_{newuserid}")

#         # Return to home with success message
#         names, rolls, times, l = extract_attendance()
#         return render_template('home.html', names=names, rolls=rolls, times=times,
#                                l=l, totalreg=totalreg(), datetoday2=datetoday2,
#                                mess=f"User '{newusername}' added, model trained, attendance marked.")

#     # GET request: show registration form
#     return render_template('register.html', message="Enter username and user ID to register")


# @app.route('/capture_face', methods=['POST'])
# def capture_face():
#     global capture_user, capture_count, capture_limit, user_folder

#     if not capture_user or capture_count >= capture_limit:
#         return "Capture finished or user not initialized.", 400

#     cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
#     time.sleep(1)
#     ret, frame = cap.read()
#     cap.release()
#     if not ret:
#         return "Failed to capture frame", 500

#     faces = extract_faces(frame)
#     for (x, y, w, h) in faces:
#         face_img = frame[y:y+h, x:x+w]
#         cv2.imwrite(f'{user_folder}/{capture_user[0]}_{capture_count}.jpg', face_img)
#         capture_count += 1
#         break  # capture one face per button click

#     if capture_count >= capture_limit:
#         train_model()
#         return "Face registration complete and model trained!", 200

#     return f"Captured {capture_count}/{capture_limit} faces", 200



# if __name__ == '__main__':
#     app.run(debug=True, port=8080)

import time
from flask import Flask, render_template, request, Response, flash, redirect, url_for, session
import cv2
import os, cv2, joblib, pandas as pd, numpy as np, csv
from datetime import date, datetime
from sklearn.neighbors import KNeighborsClassifier

# -------------------------------------------------------
# APP SETUP
# -------------------------------------------------------
app = Flask(__name__)
app.secret_key = "your_secret_key_here"

MESSAGE = "WELCOME - Instruction: To register your attendance kindly click on 'a' on the keyboard"
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

capture_user = None
capture_count = 0
capture_limit = 50
user_folder = None

os.makedirs("Attendance", exist_ok=True)
os.makedirs("static/faces", exist_ok=True)

if not os.path.exists(f"Attendance/Attendance-{datetoday}.csv"):
    with open(f"Attendance/Attendance-{datetoday}.csv", "w") as f:
        f.write("Name,Roll,Time")


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------
def totalreg():
    return len(os.listdir('static/faces'))


def extract_faces(img):
    if img is None or img.size == 0:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_detector.detectMultiScale(gray, 1.3, 5)


def identify_face(facearray):
    model = joblib.load("static/face_recognition_model.pkl")
    return model.predict(facearray)


def train_model():
    faces = []
    labels = []
    for user in os.listdir("static/faces"):
        for imgname in os.listdir(f"static/faces/{user}"):
            img = cv2.imread(f"static/faces/{user}/{imgname}")
            if img is not None:
                resized = cv2.resize(img, (50, 50))
                faces.append(resized.ravel())
                labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, "static/face_recognition_model.pkl")


def extract_attendance():
    df = pd.read_csv(f"Attendance/Attendance-{datetoday}.csv")
    return df['Name'], df['Roll'], df['Time'], len(df)


def add_attendance(name):
    if "_" not in name:
        print("Invalid name format")
        return
    username, userid = name.rsplit("_", 1)
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f"Attendance/Attendance-{datetoday}.csv")

    if str(userid) not in df["Roll"].astype(str).values:
        with open(f"Attendance/Attendance-{datetoday}.csv", "a") as f:
            f.write(f"\n{username},{userid},{current_time}")


# -------------------------------------------------------
# ADMIN AUTH
# -------------------------------------------------------
def load_admins():
    admins = []
    if os.path.exists("admin_users.csv"):
        with open("admin_users.csv", "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    admins.append((parts[0], parts[1]))
    return admins


def save_admin(username, password):
    with open("admin_users.csv", "a") as f:
        f.write(f"{username},{password}\n")


@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        admins = load_admins()

        for user, pwd in admins:
            if user == username and pwd == password:
                session["admin"] = username
                return redirect("/admin-dashboard")

        flash("Invalid username or password!", "danger")
        return redirect("/admin-login")

    return render_template("adminlogin.html")


@app.route('/admin-logout')
def admin_logout():
    session.pop("admin", None)
    return redirect("/admin-login")


@app.route('/add-admin', methods=['POST'])
def add_admin():
    if "admin" not in session:
        return redirect("/admin-login")

    username = request.form.get("new_admin_username")
    password = request.form.get("new_admin_password")

    save_admin(username, password)
    flash("New admin added!", "success")

    return redirect("/admin-dashboard")


# -------------------------------------------------------
# PAGES
# -------------------------------------------------------
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template("home.html",
                           names=names, rolls=rolls, times=times,
                           l=l, totalreg=totalreg(),
                           datetoday2=datetoday2, mess=MESSAGE)


@app.route('/admin-dashboard')
def admin_dashboard():
    if "admin" not in session:
        return redirect('/admin-login')

    names, rolls, times, l = extract_attendance()
    attendance_list = [(i + 1, names[i], rolls[i], times[i]) for i in range(l)]

    return render_template("dashboard.html",
                           attendance_list=attendance_list,
                           l=l)


# -------------------------------------------------------
# FACE RECOGNITION FEEDS
# -------------------------------------------------------
identified_person_global = None


def gen_frames():
    global identified_person_global
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    time.sleep(2)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person_global = identify_face(face.reshape(1, -1))[0]

            cv2.putText(frame, identified_person_global, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# -------------------------------------------------------
# START ATTENDANCE
# -------------------------------------------------------
@app.route('/start', methods=['GET', 'POST'])
def start():
    global identified_person_global

    if not os.path.exists("static/face_recognition_model.pkl"):
        msg = "Face not registered. Please add yourself first."
        names, rolls, times, l = extract_attendance()
        return render_template("home.html",
                               names=names, rolls=rolls, times=times,
                               l=l, totalreg=totalreg(),
                               datetoday2=datetoday2, mess=msg)

    if request.method == "POST":
        if identified_person_global:
            add_attendance(identified_person_global)
            msg = f"Attendance marked for {identified_person_global}"
        else:
            msg = "No face detected yet."

        names, rolls, times, l = extract_attendance()
        attendance_list = [(i+1, names[i], rolls[i], times[i]) for i in range(l)]

        return render_template("home.html",
                               attendance_list=attendance_list,
                               l=l,
                               totalreg=totalreg(),
                               datetoday2=datetoday2,
                               mess=msg)

    names, rolls, times, l = extract_attendance()
    attendance_list = [(i+1, names[i], rolls[i], times[i]) for i in range(l)]

    return render_template("home.html",
                           attendance_list=attendance_list,
                           l=l,
                           totalreg=totalreg(),
                           datetoday2=datetoday2)


# -------------------------------------------------------
# USER REGISTRATION
# -------------------------------------------------------
def gen_registration_frames():
    global capture_user, capture_count, user_folder
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    time.sleep(2)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,
                        f"Images: {capture_count}/{capture_limit}",
                        (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 20), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")


@app.route('/registration_video_feed')
def registration_video_feed():
    return Response(gen_registration_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add-attendance', methods=['POST'])
def add_attendance_route():
    if 'admin' not in session:
        flash("Please login first", "warning")
        return redirect('/admin-login')

    name = request.form['name']
    roll = request.form['roll']

    import cv2
    import numpy as np
    import os
    import datetime

    # Initialize camera and face detector
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    data = []
    count = 0
    max_count = 50

    # Create user folder in faces directory
    user_folder = f'static/faces/{name}_{roll}'
    os.makedirs(user_folder, exist_ok=True)

    while count < max_count:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]

            # Save face as .jpg
            cv2.imwrite(f'{user_folder}/{name}_{count}.jpg', face_img)

            # Resize and flatten for numpy array
            face_resized = cv2.resize(face_img, (50, 50))
            data.append(face_resized.flatten())

            count += 1
            break  # capture one face per frame

    cap.release()

    # Convert to numpy array and save as .npy
    data = np.array(data)
    labels = np.array([[f"{name}_{roll}"]] * data.shape[0])
    data = np.concatenate((data, labels), axis=1)

    os.makedirs("static/face_data", exist_ok=True)
    np.save(f"static/face_data/{name}_{roll}.npy", data)

    # Record attendance
    with open("Attendance.csv", "a") as f:
        f.write(f"{name},{roll},{datetime.datetime.now()}\n")

    flash("Attendance added successfully!", "success")
    return redirect('/admin-dashboard')

@app.route('/clean-attendance')
def clean_attendance():
    import pandas as pd
    import datetime

    df = pd.read_csv("Attendance.csv", header=None, names=["Name", "Roll", "Datetime"])

    df["Date"] = pd.to_datetime(df["Datetime"]).dt.date

    df_clean = df.drop_duplicates(subset=["Name", "Roll", "Date"], keep="first")

    df_clean[["Name", "Roll", "Datetime"]].to_csv("Attendance.csv", index=False, header=False)

    return "Attendance cleaned successfully!"


@app.route('/capture_attendance', methods=['GET', 'POST'])
def capture_attendance():
    if 'admin' not in session:
        flash("Please login first", "warning")
        return redirect('/admin-login')

    if request.method == "POST":
        newusername = request.form["name"]
        newuserid = request.form["roll"]

        global capture_user, capture_count, user_folder
        capture_user = (newusername, newuserid)
        user_folder = f"static/faces/{newusername}_{newuserid}"
        os.makedirs(user_folder, exist_ok=True)
        capture_count = 0

        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        time.sleep(1)
        while capture_count < 50:
            ret, frame = cap.read()
            if not ret:
                continue
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                face_img = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                cv2.imwrite(f"{user_folder}/{newusername}_{capture_count}.jpg", face_img)
                capture_count += 1
                break
        cap.release()

        train_model()
        add_attendance(f"{newusername}_{newuserid}")
        flash(f"Attendance for {newusername} added successfully!", "success")
        return redirect('/admin-dashboard')

    return render_template('capture_attendance.html')




@app.route('/add', methods=['GET', 'POST'])
def add_user():
    global capture_user, capture_count, user_folder

    if request.method == "POST":
        newusername = request.form["newusername"]
        newuserid = request.form["newuserid"]

        capture_user = (newusername, newuserid)
        user_folder = f"static/faces/{newusername}_{newuserid}"
        os.makedirs(user_folder, exist_ok=True)

        capture_count = 0

        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        time.sleep(1)

        while capture_count < capture_limit:
            ret, frame = cap.read()
            if not ret:
                continue

            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                cv2.imwrite(f"{user_folder}/{newusername}_{capture_count}.jpg", face_img)
                capture_count += 1
                break

        cap.release()

        train_model()
        add_attendance(f"{newusername}_{newuserid}")

        names, rolls, times, l = extract_attendance()

        return render_template("home.html",
                               names=names,
                               rolls=rolls,
                               times=times,
                               l=l,
                               totalreg=totalreg(),
                               datetoday2=datetoday2,
                               mess=f"User '{newusername}' added successfully!")

    return render_template("register.html",
                           message="Enter username and user ID to register")


# -------------------------------------------------------
# RUN APP
# -------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8080)
