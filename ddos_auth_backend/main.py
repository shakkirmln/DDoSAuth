# Import all the necessary files!
from tensorflow.compat.v1.keras import layers, Model
from tensorflow.compat.v1.keras.backend import set_session
from datetime import datetime
import base64
from flask_cors import CORS
from flask import jsonify
import numpy as np
import json
import cv2
from flask import Flask, request
import os
import tensorflow.compat.v1 as tf
import speech_recognition as sr
tf.disable_v2_behavior()

graph = tf.get_default_graph()

app = Flask(__name__)
CORS(app)
sess = tf.Session()
set_session(sess)

model = tf.keras.models.load_model('facenet_keras.h5')
face_model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'weights.caffemodel')
r = sr.Recognizer()

# model.summary()


def img_to_encoding(path, model):
    img1 = cv2.imread(path, 1)
    # Face extraction
    (h, w) = img1.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(
        img1, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_model.setInput(blob)
    detections = face_model.forward()
    count = 0
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        confidence = detections[0, 0, i, 2]
        # If confidence > 0.5, face is detected
        if (confidence > 0.5):
            count += 1
            frame = img1[startY:endY, startX:endX]
            img1 = frame
    if count == 0 or count > 1:
        return []
    img = img1[..., ::-1]
    dim = (160, 160)
    # resize image
    if(img.shape != (160, 160, 3)):
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    x_train = np.array([img])
    print(x_train)
    embedding = model.predict(x_train)
    print("embedding", embedding)
    return embedding


database = {}


@app.route('/captcha', methods=['POST'])
def captcha():
    file = request.files['audio_file']
    captcha = request.form['captcha']

    file.save(os.path.abspath(f'audios/recording.wav'))

    with sr.WavFile(os.path.abspath(f'audios/recording.wav')) as source:
        audio_text = r.listen(source)

    try:
        # using google speech recognition
        text = r.recognize_google(audio_text)
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
    try:
        username = request.get_json()['username']
        img_data = request.get_json()['image64']
        with open('images/'+username+'.jpg', "wb") as fh:
            fh.write(base64.b64decode(img_data[22:]))
        path = 'images/'+username+'.jpg'

        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            encoding = img_to_encoding(path, model)
            if len(encoding) == 0:
                return json.dumps({"status": 500, "msg": "Either no face detected or more than one face detected!"})
            database[username] = encoding
        return json.dumps({"status": 200})
    except:
        return json.dumps({"status": 500})


def who_is_it(image_path, database, model):
    print(image_path)
    encoding = img_to_encoding(image_path, model)
    if len(encoding) == 0:
        return -1, -1
    min_dist = 1000
    identity = None
    if len(database) == 0:
        print("No one is registered in the database!")
    else:
        # Loop over the database dictionary's names and encodings.
        for (name, db_enc) in database.items():
            dist = np.linalg.norm(encoding-db_enc)
            print(dist)
            if dist < min_dist:
                min_dist = dist
                identity = name
        if min_dist > 5:
            print("Not in the database.")
        else:
            print("it's " + str(identity) + ", the distance is " + str(min_dist))
    return min_dist, identity


@app.route('/login', methods=['POST'])
def login():
    img_data = request.get_json()['image64']
    img_name = str(int(datetime.timestamp(datetime.now())))
    with open('images/'+img_name+'.jpg', "wb") as fh:
        fh.write(base64.b64decode(img_data[22:]))
    path = 'images/'+img_name+'.jpg'
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        min_dist, identity = who_is_it(path, database, model)
    os.remove(path)
    if min_dist == -1 and identity == -1:
        return json.dumps({"msg": "Either no face detected or more than one face detected"})
    if min_dist > 5:
        return json.dumps({"identity": 0})
    return json.dumps({"identity": str(identity)})


if __name__ == "__main__":
    app.run(debug=True)
