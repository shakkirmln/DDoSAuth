import os
import cv2
import pickle
import argparse
import numpy as np
import tensorflow as tf
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-d', '--dataset', required=True,
                        help='Dataset path')

data_path = arg_parser.parse_args().dataset
model_path = 'face_authenticity.model'
le_path = 'label_encoder.pickle'

# extract image paths from dataset
imagePaths = [os.path.join(path, name) for path, subdirs,
              files in os.walk(data_path) for name in files]

data = list()
labels = list()


def get_data():
    global data, labels
    # iterate over all image paths
    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        # using 32x32 input shape for the model
        image = cv2.resize(image, (32, 32))
        data.append(image)
        labels.append(label)

    # convert data to numpy arrays
    data = np.array(data, dtype='float') / 255.0


def get_model(width, height, depth, classes):
    # initialize the model along with the input shape to be
    # 'channels last' and the channels dimension itself
    INPUT_SHAPE = (height, width, depth)
    chanDim = -1  # use for batch normalization along axis

    # if we are using "channels first", update the input shape
    # and channels dimension
    # note that: normally, by default, it's "channels last"
    if tf.keras.backend.image_data_format() == 'channels_first':
        INPUT_SHAPE = (depth, height, width)
        chanDim = 1

    model = tf.keras.Sequential([
        # Batch 1: CONV => BatchNorm CONV => BatchNorm => MaxPool => Dropout
        tf.keras.layers.Conv2D(filters=16, kernel_size=(
            3, 3), padding='same', activation='relu', input_shape=INPUT_SHAPE),
        tf.keras.layers.BatchNormalization(axis=chanDim),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(
            3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(axis=chanDim),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Batch 2: CONV => BatchNorm CONV => BatchNorm => MaxPool => Dropout
        tf.keras.layers.Conv2D(filters=32, kernel_size=(
            3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(axis=chanDim),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(
            3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(axis=chanDim),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Batch 3: FullyConnected => BatchNorm => Dropout => FullyConnected Output
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(classes, activation='softmax')
    ])

    return model


def train_model(labels):
    # encoding the labels from (fake, real) to  (0,1)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = tf.keras.utils.to_categorical(labels, 2)

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.20, random_state=42)
    aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20,
                                                          zoom_range=0.15,
                                                          width_shift_range=0.2,
                                                          height_shift_range=0.2,
                                                          shear_range=0.15,
                                                          horizontal_flip=True,
                                                          fill_mode='nearest')
    INIT_LR = 1e-4
    BATCH_SIZE = 4
    EPOCHS = 50

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
    model = get_model(width=32, height=32, depth=3, classes=len(le.classes_))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    model.fit(x=aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
              validation_data=(X_test, y_test),
              steps_per_epoch=len(X_train) // BATCH_SIZE,
              epochs=EPOCHS)
    predictions = model.predict(x=X_test, batch_size=BATCH_SIZE)

    print(classification_report(y_test.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=le.classes_))

    # saving the model and label encoder
    model.save(model_path, save_format='h5')
    with open(le_path, 'wb') as file:
        file.write(pickle.dumps(le))


get_data()
train_model(labels)
