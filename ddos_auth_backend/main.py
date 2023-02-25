from tensorflow.compat.v1.keras.backend import set_session
from flask import Flask, Response, request, jsonify
from scipy.spatial import distance as dist
from imutils import face_utils
from flask_cors import CORS
import tensorflow.compat.v1 as tf
import speech_recognition as sr
import face_recognition
import numpy as np
import threading
import imutils
import pickle
import json
import math
import dlib
import cv2
import os

tf.disable_v2_behavior()

graph = tf.get_default_graph()

app = Flask(__name__)
CORS(app)
sess = tf.Session()
set_session(sess)

STREAM_CLOSE = False
AUDIO_STREAM_RUNNING = False
BLINK_COUNT_STOP = False
LIP_MOVEMENT_AUDIO_STREAM = False

LIP_MOVEMENT = False
BLINK_COUNTER = 0
BLINK_TOTAL = 0
FACE_COUNT = 0
FAKE_FACE_COUNT = 0

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

# extacting eyes and mouth coordinates
(leftEyeStart, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightEyeStart, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mouthStart, mouthEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# initialize a lock used to ensure thread-safe
# exchanges of the frames (useful for multiple browsers/tabs
# are viewing the stream)
lock = threading.Lock()

face_detection_model = cv2.dnn.readNetFromCaffe(
    'models/face_detector/deploy.prototxt', 'models/face_detector/res10_300x300_ssd_iter_140000.caffemodel')
face_authentication_model = tf.keras.models.load_model(
    'models/face_authenticity_detector/face_authenticity.model')
face_landmark_model = dlib.shape_predictor(
    'models/face_landmark_detector/shape_predictor_68_face_landmarks.dat')
face_identification_model = tf.keras.models.load_model(
    'models/face_identifier/facenet_keras.h5')
speech_recognizer_model = sr.Recognizer()

face_authentication_label_encoder = pickle.loads(
    open('models/face_authenticity_detector/label_encoder.pickle', 'rb').read())

database = {}

extracted_face = None


def generate_video_frame():
    # grab global references to the lock variable
    global lock
    global sess
    global graph
    global FACE_COUNT
    global FAKE_FACE_COUNT
    global extracted_face
    # initialize the video stream
    vc = cv2.VideoCapture(0)

    # check camera is open
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    # while streaming
    while rval:
        # wait until the lock is acquired
        with lock:
            if STREAM_CLOSE:
                break
            face_count = 0
            fake_face_count = 0
            # read next frame
            rval, frame = vc.read()
            # if blank frame
            if frame is None:
                continue

            frame = imutils.resize(frame, width=800)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(
                frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

            face_detection_model.setInput(blob)
            detections = face_detection_model.forward()

            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e. probability) associated with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > 0.5:
                    face_count += 1
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype('int')

                    dlibRect = dlib.rectangle(startX, startY, endX, endY)

                    startX = max(0, startX-20)
                    startY = max(0, startY-20)
                    endX = min(w, endX+20)
                    endY = min(h, endY+20)

                    # extract the face ROI
                    face = frame[startY:endY, startX:endX]
                    extracted_face = face

                    with graph.as_default():
                        set_session(sess)
                        (authenticity_label,
                         prediction) = face_authenticity_detection(face)

                    if authenticity_label == 'No face detected':
                        break

                    if authenticity_label == 'real':
                        # detect facial landmarks
                        shape = face_landmark_model(gray, dlibRect)
                        lip_movement_detection(shape, frame)
                        if not BLINK_COUNT_STOP:
                            blink_detection(shape, frame)
                    elif authenticity_label == 'fake':
                        fake_face_count += 1

                    visualize_labels(frame, authenticity_label,
                                     prediction, startX, startY, endX, endY)

            FACE_COUNT = face_count
            FAKE_FACE_COUNT = fake_face_count

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", frame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
    # release the camera
    vc.release()


def face_identification_embedding(extracted_face):
    extracted_face = extracted_face[..., ::-1]
    dim = (160, 160)
    # resize image
    if(extracted_face.shape != (160, 160, 3)):
        extracted_face = cv2.resize(
            extracted_face, dim, interpolation=cv2.INTER_AREA)
    face_data = np.array([extracted_face])
    embedding = face_identification_model.predict(face_data)
    return embedding


def get_lip_height(lip):
    sum = 0
    for i in [2, 3, 4]:
        # distance between two near points up and down
        distance = math.sqrt((lip[i][0] - lip[12-i][0])**2 +
                             (lip[i][1] - lip[12-i][1])**2)
        sum += distance
    return sum / 3


def get_mouth_height(top_lip, bottom_lip):
    sum = 0
    for i in [8, 9, 10]:
        # distance between two near points up and down
        distance = math.sqrt((top_lip[i][0] - bottom_lip[18-i][0])**2 +
                             (top_lip[i][1] - bottom_lip[18-i][1])**2)
        sum += distance
    return sum / 3


def lip_movement_detection(shape, frame):
    global LIP_MOVEMENT, LIP_MOVEMENT_AUDIO_STREAM
    shape = face_utils.shape_to_np(shape)

    top_lip = [tuple(e) for e in shape[48:56]] + list(reversed([tuple(e)
                                                                for e in shape[60:64]]))
    bottom_lip = [tuple(e) for e in shape[54:62]] + list(reversed([tuple(e)
                                                                   for e in shape[64:69]]))

    top_lip_height = get_lip_height(top_lip)
    bottom_lip_height = get_lip_height(bottom_lip)
    mouth_height = get_mouth_height(top_lip, bottom_lip)

    if mouth_height > min(top_lip_height, bottom_lip_height) * 0.5:
        LIP_MOVEMENT = True
        if AUDIO_STREAM_RUNNING:
            LIP_MOVEMENT_AUDIO_STREAM = True
    else:
        LIP_MOVEMENT = False

    # visualize mouth position
    mouth = shape[mouthStart:mouthEnd]
    mouthHull = cv2.convexHull(mouth)
    cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)


def blink_detection(shape, frame):
    global BLINK_COUNTER, BLINK_TOTAL
    shape = face_utils.shape_to_np(shape)

    # extract the left and right eye coordinate
    leftEye = shape[leftEyeStart:leftEyeEnd]
    rightEye = shape[rightEyeStart:rightEyeEnd]

    # compute the eye aspect ratio for both eyes
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    # average the eye aspect ratio together for both eyes
    ear = (leftEAR + rightEAR) / 2.0

    # check to see if the eye aspect ratio is below the blink
    # threshold, and if so, increment the blink frame counter
    if ear < EYE_AR_THRESH:
        BLINK_COUNTER += 1

    # otherwise, the eye aspect ratio is not below the blink
    # threshold
    else:
        # if the eyes were closed for a sufficient number of
        # then increment the total number of blinks
        if BLINK_COUNTER >= EYE_AR_CONSEC_FRAMES:
            BLINK_TOTAL += 1

        # reset the eye frame counter
        BLINK_COUNTER = 0

    # visualize eye position
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


def face_authenticity_detection(face):
    try:
        face = cv2.resize(face, (32, 32))
    except:
        return ("No face detected", 0)
    face = face.astype('float') / 255.0
    face = tf.keras.preprocessing.image.img_to_array(face)

    # tf model require batch of data to feed in
    # so if we need only one image at a time, we have to add one more dimension
    # in this case it's the same with [face]
    face = np.expand_dims(face, axis=0)

    # pass the face ROI through the trained liveness detection model
    # to determine if the face is 'real' or 'fake'
    # predict return 2 value for each example (because in the model we have 2 output classes)
    # the first value stores the prob of being real, the second value stores the prob of being fake
    # so argmax will pick the one with highest prob
    # we care only first output (since we have only 1 input)
    preds = face_authentication_model.predict(face)[0]
    j = np.argmax(preds)
    label_name = face_authentication_label_encoder.classes_[j]

    return (label_name, preds[j])


def visualize_labels(frame, authenticity_label, prediction, startX, startY, endX, endY):
    if authenticity_label == 'real':
        top_label = f'{authenticity_label}: {prediction:.4f} Blinks: {BLINK_TOTAL}'
        bottom_label = f'Lip Movement: {LIP_MOVEMENT}'
    else:
        top_label = f'{authenticity_label}: {prediction:.4f}'
        bottom_label = ''

    # draw the label and bounding box on the frame
    cv2.putText(frame, top_label, (startX, startY - 10),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, bottom_label, (startX, endY + 25),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
    cv2.rectangle(frame, (startX, startY),
                  (endX, endY), (0, 0, 255), 4)


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


def identify_face(face_embedding):
    min_dist = 1000
    identity = None
    if len(database) == 0:
        print("No one is registered in the database!")
    else:
        # Loop over the database dictionary's names and encodings.
        for (name, db_enc) in database.items():
            dist = np.linalg.norm(face_embedding-db_enc)
            print(dist)
            if dist < min_dist:
                min_dist = dist
                identity = name
        if min_dist > 5:
            identity = ""
            print("Not in the database.")
        else:
            print("it's " + str(identity) + ", the distance is " + str(min_dist))
    return identity


@app.route('/stop_blink_count', methods=['GET'])
def stop_blink_count():
    global BLINK_COUNT_STOP
    BLINK_COUNT_STOP = True
    return Response("Blink count stopped", mimetype="text/plain")


@app.route('/start_audio_stream', methods=['GET'])
def start_audio_stream():
    global AUDIO_STREAM_RUNNING, LIP_MOVEMENT_AUDIO_STREAM
    LIP_MOVEMENT_AUDIO_STREAM = False
    AUDIO_STREAM_RUNNING = True
    return Response("Audio Stream started", mimetype="text/plain")


@app.route('/close_audio_stream', methods=['GET'])
def close_audio_stream():
    global AUDIO_STREAM_RUNNING
    AUDIO_STREAM_RUNNING = False
    return Response("Audio Stream stopped", mimetype="text/plain")


@app.route('/close_video_feed', methods=['GET'])
def close_stream():
    global STREAM_CLOSE, BLINK_COUNT_STOP
    BLINK_COUNT_STOP = False
    STREAM_CLOSE = True
    return Response("Stream closed", mimetype="text/plain")


@app.route('/video_feed/<id>', methods=['GET'])
def stream(id):
    global LIP_MOVEMENT, LIPS_APART, BLINK_COUNTER, BLINK_TOTAL, STREAM_CLOSE, BLINK_COUNT_STOP
    STREAM_CLOSE = False
    BLINK_COUNT_STOP = False
    LIP_MOVEMENT = False
    LIPS_APART = False
    BLINK_COUNTER = 0
    BLINK_TOTAL = 0
    return Response(generate_video_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/captcha', methods=['POST'])
def captcha():
    file = request.files['audio_file']
    captcha = request.form['captcha']

    if LIP_MOVEMENT_AUDIO_STREAM == False:
        return jsonify("Lip movement not identified!")

    file.save(os.path.abspath(f'audios/recording.wav'))

    with sr.WavFile(os.path.abspath(f'audios/recording.wav')) as source:
        audio_text = speech_recognizer_model.listen(source)

    try:
        # using google speech recognition
        text = speech_recognizer_model.recognize_google(audio_text)
        print('Converting audio transcripts into text ...')
        print(text)
    except:
        print('Sorry.. run again...')
        return jsonify("Sorry, read the word again!")
    if text.lower() == captcha.lower():
        return jsonify("Speech Verified!")
    else:
        return jsonify(f'Your word "{text}" does not match the word "{captcha}"')


@app.route('/register', methods=['POST'])
def register():
    global BLINK_TOTAL, BLINK_COUNTER
    try:
        username = request.get_json()['username']
        if FACE_COUNT == 0:
            return json.dumps({"status": 500, "msg": "No face detected!"})
        elif FACE_COUNT > 1:
            return json.dumps({"status": 500, "msg": f"{FACE_COUNT} face detected! Out of which {FAKE_FACE_COUNT} are fake!"})
        elif FACE_COUNT == 1 and FAKE_FACE_COUNT > 0:
            return json.dumps({"status": 500, "msg": f"{FAKE_FACE_COUNT} fake face detected!"})

        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            face_embedding = face_identification_embedding(extracted_face)
            database[username] = face_embedding
        BLINK_TOTAL = 0
        BLINK_COUNTER = 0
        return json.dumps({"status": 200})
    except:
        BLINK_TOTAL = 0
        BLINK_COUNTER = 0
        return json.dumps({"status": 500})


@app.route('/login', methods=['POST'])
def login():
    global BLINK_TOTAL, BLINK_COUNTER, BLINK_COUNT_STOP
    try:
        target_blink_count = request.get_json()['blinkCount']
        if FACE_COUNT == 0:
            return json.dumps({"status": 500, "msg": "No face detected!"})
        elif FACE_COUNT > 1:
            return json.dumps({"status": 500, "msg": f"{FACE_COUNT} face detected! Out of which {FAKE_FACE_COUNT} are fake!"})
        elif FACE_COUNT == 1 and FAKE_FACE_COUNT > 0:
            return json.dumps({"status": 500, "msg": f"{FAKE_FACE_COUNT} fake face detected!"})
        elif BLINK_TOTAL != target_blink_count:
            BLINK_TOTAL = 0
            BLINK_COUNTER = 0
            BLINK_COUNT_STOP = False
            return json.dumps({"status": 500, "msg": f"Please blink exactly {target_blink_count} times!"})

        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            face_embedding = face_identification_embedding(extracted_face)
            identity = identify_face(face_embedding)
            if identity is None:
                return json.dumps({"msg": "No one is registered in the database"})
            elif identity == "":
                return json.dumps({"msg": "Not in the database"})
            BLINK_TOTAL = 0
            BLINK_COUNTER = 0
            BLINK_COUNT_STOP = False
            return json.dumps({"identity": str(identity)})
    except:
        BLINK_TOTAL = 0
        BLINK_COUNTER = 0
        BLINK_COUNT_STOP = False
        return json.dumps({"status": 500})


if __name__ == "__main__":
    app.run(debug=True)
