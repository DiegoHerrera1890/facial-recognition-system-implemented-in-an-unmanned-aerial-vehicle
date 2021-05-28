from PIL import Image
from keras.applications.vgg19 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np
from keras_vggface import utils

CATEGORIES = ["A", "B", "C", "D", "E", "F"]

from keras.preprocessing import image

model = load_model('facefeatures_model_100_epochs.h5')

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    # Crop all faces found
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cropped_face = img[y:y + h, x:x + w]

    return cropped_face


# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    # canvas = detect(gray, frame)
    # image, face =face_detector(frame)

    face = face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
        # Resizing into 128x128 because we trained the model with this image size.
        img_array = np.array(im)
        # Our keras model used a 4D tensor, (images x height x width x channel)
        # So changing dimension 128x128x3 into 1x128x128x3
        img_array = np.expand_dims(img_array, axis=0)
        print(img_array.shape)
        # img_array = utils.preprocess_input(img_array, version=2) # or version=2
        # img_array = preprocess_input(img_array, data_format=None)
        img_array = preprocess_input(img_array, data_format=None)
        pred = model.predict(img_array)
        pred_name = CATEGORIES[np.argmax(pred)]
        print(pred_name)
        print("Clase 1: ", pred[0][0])
        print("Clase 2: ", pred[0][1])
        print("Clase 3: ", pred[0][2])
        print("Clase 4: ", pred[0][3])
        print("Clase 5: ", pred[0][4])
        print("Clase 6: ", pred[0][5])

        name = "None matching"
        if (pred[0][0] > 0.4):
            name = '1   '
        if (pred[0][1] > 0.5):
            name = '2'
        if (pred[0][2] > 0.5):
            name = '3'
        if (pred[0][3] > 0.5):
            name = '4'
        if (pred[0][4] > 0.5):
            name = '5'
        cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
