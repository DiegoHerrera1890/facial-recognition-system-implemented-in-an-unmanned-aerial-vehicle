#!/usr/bin/env python3
import numpy as np
import cv2
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from time import sleep

rospy.init_node('Face_detection_node', anonymous=True)
cap = cv2.VideoCapture(0)

#frame_width = 1280 # int(cap.get(3))
#frame_height = 720 # int(cap.get(4))
# sleep(5)
rate = rospy.Rate(10)

while True:  # not rospy.is_shutd1.6wn():
    rospy.loginfo("Video is striming")
    ret, img = cap.read()
    # print(img.shape)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.resize(img, (1280, 720))
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    rate.sleep()

cap.release()
cv2.destroyAllWindows()
