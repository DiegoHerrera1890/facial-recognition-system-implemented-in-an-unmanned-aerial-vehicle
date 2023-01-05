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
pub = rospy.Publisher('/Face_recognition/Searching', String, queue_size=10)
#pub2 = rospy.Publisher('/camera_jetson/image_raw', Image, queue_size=100)
pub3 = rospy.Publisher('/Face_recognition/face_coordinates', Point, queue_size=10)
pub4 = rospy.Publisher('/Face_recognition/face_found', String, queue_size=10)
pub5 = rospy.Publisher('/Face_recognition/initial_position', Pose, queue_size=10)
face_cascade = cv2.CascadeClassifier('/home/xonapa/drone_ws/src/face_detection/scripts/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(-1) 
frame_width = 1280 #int(cap.get(3))
frame_height = 720 #int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                            # /media/jetson/B2E4-69EA1/videos
video_out = cv2.VideoWriter('/media/xonapa/B2E4-69EA/videos/4/face_detection_11_15.avi', fourcc, 15, (frame_width,frame_height))

img_count = 0
rate = rospy.Rate(10) 

while not rospy.is_shutdown():
    ret, img = cap.read()
    img_count += 1
    faces = face_cascade.detectMultiScale(img, 1.3, 9)
    detections = []
    detections2 = ()
    olap = []
    
    if len(faces) > 0:
        rospy.loginfo("Face detected")
        message_stringg = 'face_found'
        pub4.publish(message_stringg)
        font = cv2.FONT_HERSHEY_DUPLEX
        for (x, y, w, h) in faces:
            c1 =x+(w//2) # Center of BBox X
            c2 =y+(h//2) # Center of BBox Y
            A = h*w  # Area of Bounding box
            detections.append([x,y,(x+w),(y+h)])
            detections2 = np.asarray(detections)
            img_count += 1
            
        boxes_ids = tracker.update(detections2)
        
        for box_id in boxes_ids:
            x1,y1,x2,y2,id_n = box_id
            id_N = int(id_n)
            X1,Y1,X2,Y2 = int(x1),int(y1),int(x2),int(y2)
            olap.append(id_N)
            testodesu.noidea(boxes_ids,olap,X1,Y1,X2,Y2,font,img)
            
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

'''
    if ret:
        video_out.write(img)
'''    

#video_out.release()
cap.release()
cv2.destroyAllWindows()
