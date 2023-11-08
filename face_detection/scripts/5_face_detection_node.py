#!/usr/bin/env python3
import numpy as np
#import jetson.utils
import cv2
import rospy
import csv
from std_msgs.msg import String
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
import sys
from cv_bridge import CvBridge
from time import sleep
from sort import *
import testodesu

bridge = CvBridge()
tracker = Sort() 

rospy.init_node('Face_detection_node', anonymous=True)
pub5 = rospy.Publisher('/Face_recognition/initial_position', Pose, queue_size=10)
pub = rospy.Publisher('/Face_recognition/Searching', String, queue_size=10)
#pub2 = rospy.Publisher('/camera_jetson/image_raw', Image, queue_size=100)
pub_face_found = rospy.Publisher('/Face_recognition/face_found', String, queue_size=10)
pub_face_coordinates = rospy.Publisher('/Face_recognition/face_coordinates', Point, queue_size=10)

face_cascade = cv2.CascadeClassifier('/home/xonapa/drone_ws/src/face_detection/scripts/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(-1) 
frame_width = int(cap.get(3)) # 1280
frame_height = int(cap.get(4)) # 720
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# /media/jetson/B2E4-69EA1/videos
video_out = cv2.VideoWriter('/media/xonapa/B2E4-69EA5/videos/4/face_detection_11_15.avi', fourcc, 15, (frame_width,frame_height))

img_count = 0
rate = rospy.Rate(10) 
num_photos = 10
photo_count = 0
while not rospy.is_shutdown():
    ret, img = cap.read()
    # Get the size of the image
    height, width, channels = img.shape
    rospy.loginfo("Image size: {} x {}".format(width, height))
    faces = face_cascade.detectMultiScale(img, 1.3, 9)
    
    if len(faces) > 0:
        rospy.loginfo("Face detected")
        message_stringg = 'face_found'
        pub_face_found.publish(message_stringg)
        font = cv2.FONT_HERSHEY_DUPLEX
        for (x, y, w, h) in faces:
            c1 = x + (w // 2)  # Center of BBox X
            c2 = y + (h // 2)  # Center of BBox Y
            A = h * w  # Area of Bounding box
            roi = img[y-2:y + h+2, x-2:x + w+2]
            roi = cv2.resize(roi, (96, 96))

            coordinates = Point(x=c1, y=A, z=c2)
            pub_face_coordinates.publish(coordinates)
        rospy.loginfo("Area of %f", A)
        cv2.putText(img, "Face found", (x,y - 15), font, 1, (255, 0, 0), 1)
        cv2.putText(img, str(A), (x,y - 25), font, 1, (255, 0, 0), 1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            
    else:
        rospy.loginfo("Face not found")
        message_string = 'Searching'
        pub.publish(message_string)
        
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('frame', img)

    key = cv2.waitKey(1)
    if key == 27:
        break

    rate.sleep()


cap.release()
cv2.destroyAllWindows()
