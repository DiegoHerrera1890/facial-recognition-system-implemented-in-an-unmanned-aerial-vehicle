#!/usr/bin/env python3
import numpy as np
import cv2
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from time import sleep

bridge = CvBridge()

def initial_position(i):
    # takeoff_client(True)
    while i < 2:
        print("initial position homnieeee")
        pose = Pose()
        X, Y, Z = 0, 0, 1.6
        pose.position.x = X
        pose.position.y = Y
        pose.position.z = Z
        pub5.publish(pose)
        i += 1
        sleep(4)


rospy.init_node('Face_detection_node', anonymous=True)
pub = rospy.Publisher('/Face_recognition/Searching', String, queue_size=10)
pub2 = rospy.Publisher('/Face_recognition/face_coordinates', Point, queue_size=10)
pub3 = rospy.Publisher('/Face_recognition/face_found', Point, queue_size=10)
pub4 = rospy.Publisher('/Face_recognition_s/face_found_data', String, queue_size=10)
pub5 = rospy.Publisher('/Face_recognition/initial_position', Pose, queue_size=10)
face_cascade = cv2.CascadeClassifier(
    '/home/diego/catkin_ws/src/face_detection/scripts/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_out = cv2.VideoWriter('video_1.avi', fourcc, 30, (frame_width, frame_height))
initial_position(0)
# sleep(5)
rate = rospy.Rate(10)

while True:  # not rospy.is_shutd1.6wn():
    rospy.loginfo("Video is beginning")
    ret, img = cap.read()
    # print(img.shape)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    two_pi = 2 * 3.1415926535
    if len(faces) > 0:
        print("Hay una cara homies")
        message_stringg = 'face_found'
        pub4.publish(message_stringg)
        for (x, y, w, h) in faces:
            face_detected = True
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            c1 = x + (w // 2)  # Center of BBox X
            c2 = y + (h // 2)  # Center of BBox Y
            A = h * w  # Area of Bounding box
            print("Xf: ", c1, "Yf: ", c2)
            cv2.circle(img, (c1, c2), 10, (0, 255, 0), 2)
            coordinates = Point(x=c1, y=A, z=c2)
            pub3.publish(coordinates)
            # face_found = Point(x= 1, y= 1, z= 2)
            # pub3.publish(face_found)
            # rate.sleep()
        # print("Putos todos")
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, 'No hay nada homie', (0, 20), font, 1.0, (255, 0, 0), 1)
    else:
        print("Not face found")
        # coordinates_2 = Point(x=0, y=0, z=1,6)
        message_string = 'Searching'
        pub.publish(message_string)
        rate.sleep()
        # pub2.publish(coordinates_2)
        # rate.sleep()

    cv2.circle(img, (300, 240), 10, (18, 255, 255), 2)
    cv2.circle(img, (320, 240), 10, (0, 0, 255), 2)
    cv2.circle(img, (340, 240), 10, (255, 0, 255), 2)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_out.release()
cv2.destroyAllWindows()
