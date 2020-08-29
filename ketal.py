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
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)),axis=-1, keep_dims=True)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)),axis=-1, keep_dims=True)
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
database["Toyota"] = img_to_encoding("images/gregory/gregory_1.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_2.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_3.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_4.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_5.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_6.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_7.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_8.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_9.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_10.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_11.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_12.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_13.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_14.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_15.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_16.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_17.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_18.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_19.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_20.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_21.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_22.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_23.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_24.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_25.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_26.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_27.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_28.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_29.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_30.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_31.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_32.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_33.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_34.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_35.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_36.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_37.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_38.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_39.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_40.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_41.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_42.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_43.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_44.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_45.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_46.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_47.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_48.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_49.jpg", FRmodel)
database["Toyota"] = img_to_encoding("images/gregory/gregory_50.jpg", FRmodel)
#daniel
database["daniel"] = img_to_encoding("images/daniel/daniel_1.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_2.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_3.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_4.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_5.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_6.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_7.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_8.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_9.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_10.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_11.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_12.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_13.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_14.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_15.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_16.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_17.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_18.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_19.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_20.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_21.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_22.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_23.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_24.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_25.jpg", FRmodel)
database["daniel"] = img_to_encoding("images/daniel/daniel_26.jpg", FRmodel)


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
        if min_dist > 0.8:
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
