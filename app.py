from flask import Flask, render_template, Response
import cv2
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
import numpy as np
from threading import Thread
from pygame import mixer
import time

app = Flask(__name__)

mixer.init()
sound1 = mixer.Sound('wake_up.mp3')
sound2 = mixer.Sound('alert.mp3')

alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0
YARN_FRAME = 0
alarm_start_time = None

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
YAWN_THRESH = 20
YAWN_CONSEC_FRAMES = 15

def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status or alarm_status2:
        if alarm_status:
            print('Closed eyes call')
            saying = True
            sound1.play()
            time.sleep(1)
            saying = False
        if alarm_status2:
            print('Yawn call')
            sound2.play()
            time.sleep(1)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

vs = cv2.VideoCapture(0)

def generate_frames():
    global alarm_status, alarm_status2, COUNTER, YARN_FRAME, alarm_start_time

    while True:
        success, frame = vs.read()
        if not success:
            break
        else:
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

            for (x, y, w, h) in rects:
                rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                eye = final_ear(shape)
                ear = eye[0]
                leftEye = eye[1]
                rightEye = eye[2]

                distance = lip_distance(shape)

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                lip = shape[48:60]
                cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

                if ear < EYE_AR_THRESH:
                    COUNTER += 1

                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        if not alarm_status:
                            alarm_status = True
                            alarm_start_time = time.time()
                            t = Thread(target=alarm, args=('wake up sir',))
                            t.daemon = True
                            t.start()

                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    COUNTER = 0
                    if alarm_status and alarm_start_time is not None:
                        elapsed = time.time() - alarm_start_time
                        if elapsed > 2:
                            alarm_status = False
                            alarm_start_time = None

                if distance > YAWN_THRESH:
                    YARN_FRAME += 1

                    if YARN_FRAME >= YAWN_CONSEC_FRAMES:
                        if not alarm_status2:
                            alarm_status2 = True
                            t = Thread(target=alarm, args=('take some fresh air sir',))
                            t.daemon = True
                            t.start()

                        cv2.putText(frame, "Yawn Alert", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    YARN_FRAME = 0
                    alarm_status2 = False

                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
