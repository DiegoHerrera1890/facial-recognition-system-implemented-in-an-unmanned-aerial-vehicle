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
FRmodel.summary()
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

# juan
database["juan"] = (img_to_encoding(BASE_DIR + "/images/juan/juan_1.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_2.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_3.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_4.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_5.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_6.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_7.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_8.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_9.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_10.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_11.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_12.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_13.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_14.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_15.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_16.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_17.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_18.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_19.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_20.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_21.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_22.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_23.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_24.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_25.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_26.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_27.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_28.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_29.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_30.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_31.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_32.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_33.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_34.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_35.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_36.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_37.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_38.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_39.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_40.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_41.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_42.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_43.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_44.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_45.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_46.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_47.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_48.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_49.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_50.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_51.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_52.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_53.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_54.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_55.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_56.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_57.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_58.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_59.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_60.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_61.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_62.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_63.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_64.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_65.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_66.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_67.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_68.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_69.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_70.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_71.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_72.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_73.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_74.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_75.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_76.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_77.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_78.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_79.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_80.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_81.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_82.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_83.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_84.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_85.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_86.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_87.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_88.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_89.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_90.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_91.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_92.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_93.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_94.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/juan/juan_95.jpg", FRmodel))

# Nomura
database["Nomura"] = (img_to_encoding(BASE_DIR + "/images/nomura/nomura_1.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_2.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_3.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_4.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_5.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_6.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_7.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_8.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_9.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_10.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_11.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_12.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_13.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_14.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_15.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_16.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_17.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_18.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_19.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_20.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_21.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_22.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_23.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_24.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_25.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_26.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_27.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_28.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_29.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_30.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_31.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_32.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_33.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_34.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_35.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_36.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_37.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_38.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_39.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_40.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_41.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_42.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_43.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_44.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_45.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_46.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_47.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_48.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_49.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_50.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_51.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_52.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_53.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_54.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_55.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_56.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_57.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_58.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_59.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_60.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_61.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_62.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_63.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_64.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_65.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_66.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_67.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_68.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_69.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_70.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_71.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_72.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_73.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_74.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_75.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_76.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_77.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_78.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_79.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_80.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_81.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_82.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_83.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_84.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_85.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_86.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_87.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_88.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_89.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_90.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_91.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_92.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_93.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_94.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/nomura/nomura_95.jpg", FRmodel))

# maeda
database["maeda"] = (img_to_encoding(BASE_DIR + "/images/maeda/maeda_1.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_2.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_3.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_4.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_5.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_6.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_7.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_8.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_9.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_10.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_11.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_12.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_13.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_14.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_15.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_16.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_17.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_18.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_19.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_20.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_21.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_22.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_23.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_24.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_25.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_26.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_27.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_28.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_29.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_30.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_31.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_32.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_33.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_34.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_35.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_36.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_37.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_38.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_39.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_40.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_41.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_42.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_43.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_44.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_45.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_46.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_47.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_48.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_49.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_50.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_51.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_52.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_53.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_54.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_55.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_56.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_57.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_58.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_59.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_60.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_61.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_62.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_63.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_64.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_65.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_66.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_67.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_68.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_69.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_70.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_71.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_72.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_73.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_74.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_75.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_76.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_77.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_78.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_79.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_80.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_81.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_82.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_83.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_84.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_85.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_86.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_87.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_88.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_89.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_90.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_91.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_92.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_93.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_94.jpg", FRmodel),
    img_to_encoding(BASE_DIR + "/images/maeda/maeda_95.jpg", FRmodel))

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
video_out = cv2.VideoWriter('/media/xonapa/3A3A-CE50/videos/2/face_detection_2.avi', fourcc, 8, (frame_width,frame_height))
rate = rospy.Rate(10)
flag_nf = False
flag_ff = False
id_unknown = 0
id_known = 0
#total_sum = 0
#j = 0
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
        import time
        
        if len(faceRects) > 0:
            past_time2 = time.time()
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
                min_dist, identity = who_is_it(roi, database, FRmodel)

                # rospy.loginfo("First min distance is: %f", min_dist)
                average_value.write_image_value(img_count, min_dist, identity)
                avg_val, flag, identityy = average_value.calculate_average_distance(min_dist, identity, avg_val, flag=False)
                #total_sum = total_avg
                # avg_val = round(avg_val, 2)
                avg_val = round(min_dist, 2)

                img_count += 1
                '''
                j += 1
                if j == 2:
                    j = 0
                    total_sum = 0
                '''

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

            current_time2 = time.time()
            elapsed_time2 = current_time2 - past_time2
            rospy.loginfo("Time: %f seconds.", elapsed_time)

        else:
            rospy.loginfo("Face not found...")
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, 'No faces detected', (0, 20), font, 1.0, (255, 0, 0), 1)
            search_msg = "Searching"
            pub_Searching.publish(search_msg)
        

        new_h = frame_height // 2
        new_w = frame_width // 2
        ngray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(ngray, (new_w, new_h))
        image_message = bridge.cv2_to_imgmsg(resize)
        pub_image_raw.publish(image_message)
        if ret:
            video_out.write(frame)
        #cv2.imshow('frame', frame)

        rate.sleep()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_out.release()
#cap.release()
#cv2.destroyAllWindows()
