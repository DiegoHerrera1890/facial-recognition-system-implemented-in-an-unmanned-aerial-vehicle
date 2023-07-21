#!/usr/bin/env python3
import cv2
import rospy
import average_value
from keras.models import Sequential, load_model
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.utils.vis_utils import plot_model
from keras import backend as K
import time
from sort import *
K.set_image_data_format('channels_first')
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
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
from time import sleep
import csv

bridge = CvBridge()
tracker = Sort() 


rospy.init_node('Face_recognition_node', anonymous=True) 
pub_initial_position = rospy.Publisher('/Face_recognition/initial_position', Pose, queue_size=10)     
pub_Searching = rospy.Publisher('/Face_recognition/Searching', String, queue_size=10)
pub_image_raw = rospy.Publisher('/camera_jetson/image_raw', Image, queue_size=100)
pub_face_found = rospy.Publisher('/Face_recognition/face_found', String, queue_size=10)
pub_face_coordinates = rospy.Publisher('/Face_recognition/face_coordinates', Point, queue_size=10)


#pub_face_notmatch = rospy.Publisher('/Face_recognition/face_notmatch', String, queue_size=10)
pub_model_ready = rospy.Publisher('/Face_recognition/model_ready', String, queue_size=10)
#initial_position(0)

BASE_DIR = "/home/xonapa/drone_ws/src/face_recognition_siamese/scripts"
SD_DIR = "/media/xonapa/B2E4-69EA/"
'exec(%matplotlib inline)'
'exec(%load_ext autoreload)'
'exec(%autoreload 2)'

np.set_printoptions(threshold=sys.maxsize)

rospy.loginfo(keras.__version__)

FRmodel = faceRecoModel(input_shape=(3, 96, 96))

#rospy.loginfo("Total Params:", FRmodel.count_params())


def triplet_loss(y_true, y_pred, alpha=0.2):
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
    cv2.imshow(anchor)
    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1, keepdims=True)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1, keepdims=True)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    ### END CODE HERE ###

    return loss


rospy.loginfo('Loading model')
FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
load_weights_from_FaceNet(FRmodel)
FRmodel.summary()
rospy.loginfo('Model ready, passing images through the model')
  

#plot_model(FRmodel, to_file='model.png', show_shapes=True, show_layer_names=True, expand_nested=True)
database = {}

import time

past_time = time.time()
# Diego
database["diego"] = (img_to_encoding(BASE_DIR + "/images/diego/diego_1.jpg", FRmodel), 
    img_to_encoding(BASE_DIR + "/images/diego/diego_2.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_3.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_4.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_5.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_6.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_7.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_8.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_9.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_10.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_11.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_12.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_13.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_14.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_15.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_16.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_17.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_18.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_19.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_20.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_21.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_22.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_23.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_24.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_25.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_26.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_27.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_28.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_29.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_30.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_31.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_32.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_33.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_34.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_35.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_36.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_37.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_38.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_39.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_40.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_41.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_42.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_43.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_44.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_45.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_46.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_47.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_48.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_49.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_50.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_51.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_52.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_53.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_54.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_55.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_56.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_57.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_58.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_59.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_60.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_61.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_62.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_63.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_64.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_65.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_66.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_67.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_68.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_69.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_70.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_71.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_72.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_73.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_74.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_75.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_76.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_77.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_78.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_79.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_80.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_81.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_82.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_83.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_84.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_85.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_86.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_87.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_88.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_89.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_90.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_91.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_92.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_93.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_94.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/diego/diego_95.jpg", FRmodel))

# Monieer
database["Monieer"] = (img_to_encoding(BASE_DIR + "/images/monieer/monieer_1.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_2.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_3.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_4.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_5.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_6.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_7.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_8.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_9.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_10.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_11.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_12.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_13.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_14.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_15.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_16.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_17.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_18.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_19.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_20.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_21.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_22.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_23.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_24.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_25.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_26.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_27.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_28.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_29.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_30.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_31.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_32.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_33.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_34.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_35.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_36.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_37.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_38.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_39.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_40.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_41.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_42.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_43.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_44.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_45.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_46.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_47.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_48.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_49.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_50.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_51.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_52.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_53.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_54.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_55.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_56.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_57.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_58.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_59.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_60.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_61.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_62.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_63.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_64.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_65.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_66.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_67.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_68.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_69.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_70.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_71.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_72.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_73.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_74.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_75.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_76.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_77.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_78.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_79.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_80.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_81.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_82.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_83.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_84.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_85.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_86.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_87.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_88.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_89.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_90.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_91.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_92.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_93.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_94.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/monieer/monieer_95.jpg", FRmodel))

# Yamato
database["Yamato"] = (img_to_encoding(BASE_DIR + "/images/yamato/yamato_1.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_2.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_3.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_4.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_5.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_6.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_7.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_8.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_9.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_10.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_11.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_12.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_13.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_14.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_15.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_16.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_17.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_18.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_19.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_20.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_21.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_22.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_23.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_24.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_25.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_26.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_27.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_28.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_29.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_30.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_31.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_32.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_33.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_34.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_35.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_36.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_37.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_38.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_39.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_40.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_41.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_42.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_43.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_44.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_45.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_46.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_47.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_48.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_49.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_50.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_51.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_52.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_53.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_54.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_55.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_56.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_57.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_58.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_59.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_60.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_61.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_62.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_63.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_64.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_65.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_66.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_67.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_68.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_69.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_70.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_71.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_72.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_73.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_74.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_75.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_76.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_77.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_78.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_79.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_80.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_81.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_82.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_83.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_84.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_85.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_86.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_87.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_88.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_89.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_90.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_91.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_92.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_93.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_94.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/yamato/yamato_95.jpg", FRmodel))

# Karen
database["Karen"] = (img_to_encoding(BASE_DIR + "/images/karen/karen_1.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_2.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_3.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_4.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_5.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_6.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_7.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_8.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_9.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_10.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_11.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_12.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_13.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_14.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_15.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_16.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_17.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_18.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_19.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_20.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_21.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_22.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_23.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_24.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_25.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_26.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_27.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_28.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_29.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_30.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_31.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_32.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_33.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_34.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_35.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_36.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_37.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_38.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_39.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_40.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_41.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_42.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_43.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_44.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_45.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_46.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_47.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_48.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_49.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_50.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_51.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_52.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_53.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_54.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_55.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_56.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_57.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_58.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_59.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_60.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_61.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_62.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_63.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_64.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_65.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_66.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_67.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_68.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_69.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_70.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_71.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_72.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_73.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_74.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_75.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_76.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_77.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_78.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_79.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_80.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_81.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_82.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_83.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_84.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_85.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_86.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_87.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_88.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_89.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_90.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_91.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_92.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_93.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_94.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/karen/karen_95.jpg", FRmodel))

# Hibiki
database["Hibiki"] = (img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_1.jpg", FRmodel), 
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_2.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_3.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_4.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_5.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_6.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_7.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_8.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_9.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_10.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_11.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_12.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_13.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_14.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_15.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_16.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_17.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_18.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_19.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_20.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_21.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_22.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_23.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_24.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_25.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_26.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_27.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_28.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_29.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_30.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_31.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_32.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_33.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_34.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_35.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_36.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_37.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_38.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_39.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_40.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_41.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_42.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_43.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_44.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_45.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_46.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_47.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_48.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_49.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_50.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_51.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_52.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_53.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_54.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_55.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_56.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_57.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_58.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_59.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_60.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_61.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_62.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_63.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_64.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_65.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_66.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_67.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_68.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_69.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_70.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_71.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_72.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_73.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_74.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_75.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_76.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_77.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_78.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_79.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_80.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_81.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_82.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_83.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_84.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_85.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_86.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_87.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_88.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_89.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_90.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_91.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_92.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_93.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_94.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/hibiki/hibiki_95.jpg", FRmodel))

current_time = time.time()
elapsed_time = current_time - past_time
rospy.loginfo("Done with %f seconds.", elapsed_time)
rospy.loginfo("Done...")
sleep(3)
message_model = 'done'
pub_model_ready.publish(message_model)

face_cascade = cv2.CascadeClassifier(BASE_DIR + '/haarcascade_frontalface_default.xml')
faces = list(database.keys())
cap = cv2.VideoCapture(-1)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
img_count = 0
#path = r'/home/jetson/catkin_ws/src/face_recognition_keras/scripts/images/diego/diego_1.jpg'
video_out = cv2.VideoWriter('/media/xonapa/3A3A-CE50/videos/1/face_detection.avi', fourcc, 8, (frame_width,frame_height))
rate = rospy.Rate(10)
flag_nf = False
flag_ff = False
id_unknown = 0
id_known = 0
total_sum = 0
j = 0
avg_val = 1000
A_dict = {'': 0}
B_dict = {'': 0}
name = ''

while not rospy.is_shutdown():
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faceRects = face_cascade.detectMultiScale(gray, 1.3, 5)
        detections = []
        detections2 = ()
        olap = []

        if len(faceRects) > 0:
            rospy.loginfo("Face found...")
            face_message = "face_found"
            pub_face_found.publish(face_message)

            for (x, y, w, h) in faceRects:
                c1 = x + (w // 2)  # Center of BBox X
                c2 = y + (h // 2)  # Center of BBox Y
                A = h * w  # Area of Bounding box
                coordinates = Point(x=c1, y=A, z=c2)
                roi = frame[y-2:y + h+2, x-2:x + w+2]
                roi = cv2.resize(roi, (96, 96))
                #roi2 = cv2.imread(path)
                min_dist, identity = who_is_it(roi, database, FRmodel)

                average_value.write_image(img_count, roi)
                average_value.write_image_value(img_count, min_dist, identity)
                avg_val, flag, total_avg, identityy = average_value.calculate_average_distance(j, min_dist, total_sum, identity, avg_val, flag=False)
                total_sum = total_avg
                avg_val = round(avg_val, 2)

                img_count += 1
                j += 1
                if j == 2:
                    j = 0
                    total_sum = 0

                if img_count == 80:
                    img_count = 0

                detections.append([x, y, (x+w), (y+h)])
                detections2 = np.asarray(detections)

            boxes_ids = tracker.update(detections2)

            for box_id in boxes_ids:
                x1, y1, x2, y2, id_n = box_id
                id_N = int(id_n) # id otorgado por SORT en numeros desde numero 1
                # X1,Y1,X2,Y2 = int(x1),int(y1),int(x2),int(y2)
                olap.append(id_N)

                if len(boxes_ids) > 1 and len(olap) == len(boxes_ids):
                    id_N = olap[len(olap) - 1]
                    X1, Y1, X2, Y2 = int(x1), int(y1), int(x2), int(y2)
                    average_value.new_algorithm(coordinates, X1, Y1, X2, Y2, flag, avg_val, id_N, A_dict, B_dict, identityy, name, frame, id_known, id_unknown, flag_nf, flag_ff)

                elif len(boxes_ids) == 1:
                    id_N = olap[len(olap) - 1]
                    X1, Y1, X2, Y2 = int(x1), int(y1), int(x2), int(y2)
                    average_value.new_algorithm(coordinates, X1, Y1, X2, Y2, flag, avg_val, id_N, A_dict, B_dict, identityy, name, frame, id_known, id_unknown, flag_nf, flag_ff)

        else:
            rospy.loginfo("Face not found...")
            #flag_nf = False
            #flag_ff = False
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, 'No faces detected', (0, 20), font, 1.0, (255, 0, 0), 1)
            search_msg = "Searching"
            pub_Searching.publish(search_msg)

        new_h = frame_height // 2
        new_w = frame_width // 2
        resize = cv2.resize(gray, (new_w, new_h))
        image_message = bridge.cv2_to_imgmsg(resize)
        pub_image_raw.publish(image_message)
        if ret:
            video_out.write(frame)
        # cv2.imshow('frame', frame)

        rate.sleep()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_out.release()
# cap.release()
# cv2.destroyAllWindows()

