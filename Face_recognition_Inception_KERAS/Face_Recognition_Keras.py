from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import sys
import keras
from imutils import paths
import imutils
import pickle
from face_rec_model import who_is_it
from face_rec_model import who_is_it_image
from face_rec_model import verify
from face_rec_model import img_to_encoding
from face_rec_model import img_to_encoding_1
import face_recognition
'exec(%matplotlib inline)'
'exec(%load_ext autoreload)'
'exec(%autoreload 2)'

np.set_printoptions(threshold=sys.maxsize)


print(keras.__version__)
FRmodel = faceRecoModel(input_shape=(3, 96, 96))

print("Total Params:", FRmodel.count_params())

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)),axis=-1, keepdims=True)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)),axis=-1, keepdims=True)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    ### END CODE HERE ###
    
    return loss

print('Loading model')
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)
print('Model ready')


database = {}
#face_encodings(face_image, known_face_locations=None, num_jitters=1):
#レオン

#Diego
database["diego"] = img_to_encoding("images/diego/diego_1.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_2.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_3.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_4.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_5.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_6.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_7.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_8.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_9.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_10.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_11.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_12.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_13.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_14.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_15.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_16.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_17.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_18.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_19.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_20.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_21.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_22.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_23.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_24.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_25.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_26.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_27.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_28.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_29.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_30.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_31.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_32.jpg", FRmodel)
database["diego"] = img_to_encoding("images/diego/diego_33.jpg", FRmodel)

#Greg
database["Monieer"] = img_to_encoding("images/monieer/monieer_1.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_2.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_3.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_4.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_5.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_6.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_7.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_8.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_9.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_10.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_11.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_12.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_13.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_14.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_15.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_16.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_17.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_18.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_19.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_20.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_21.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_22.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_23.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_24.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_25.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_26.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_27.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_28.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_29.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_30.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_31.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_32.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_33.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_34.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_35.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_36.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_37.jpg", FRmodel)
database["Monieer"] = img_to_encoding("images/monieer/monieer_38.jpg", FRmodel)

#daniel
database["Yamato"] = img_to_encoding("images/yamato/yamato_1.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_2.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_3.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_4.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_5.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_6.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_7.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_8.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_9.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_10.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_11.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_12.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_13.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_14.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_15.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_16.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_17.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_18.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_19.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_20.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_21.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_22.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_23.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_24.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_25.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_26.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_27.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_28.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_29.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_30.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_31.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_32.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_33.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_34.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_35.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_36.jpg", FRmodel)
database["Yamato"] = img_to_encoding("images/yamato/yamato_37.jpg", FRmodel)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    #frame = imutils.resize(frame, width = 800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceRects = face_cascade.detectMultiScale(gray, 1.3, 5)
    #face_locations = face_recognition.face_locations(frame)    
    
    for (x, y, w, h) in faceRects:
        roi = frame[y:y+h,x:x+w]
        #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi,(96, 96))
        #min_dist = 1000
        faces = list(database.keys())
        detected  = False
        
        for face in range(len(faces)):
            person = faces[face]
                            #verify("roi", "diego", database, FRmodel)
                            #verify(image_path, identity, database, model): dist, door_open
                            #who_is_it(image_path, database, model): min_dist, identity
            #dist, detected = verify(roi, person, database[person], FRmodel)
            min_dist, identity = who_is_it(roi, database, FRmodel)
        #print(identity)
        if min_dist > 1.0:
            name = 'Unknown'
        else:
            name = identity
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
            #cv2.putText(frame, identity, (x+ (w//2),y-2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)
        cv2.putText(frame, name, (x + 6, y - 6), font, 1.0, (255, 255, 255), 1)
        #cv2.putText(frame, identity, (x+ (w//2),y-2), font, 1.0, (0, 0, 255), 1)
        '''else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            #cv2.putText(frame, identity, (x+ (w//2),y-2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)
            cv2.putText(frame, 'nO', (x + 6, y - 6), font, 1.0, (255, 255, 255), 1)'''
    cv2.imshow('frame', frame)
    #out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
