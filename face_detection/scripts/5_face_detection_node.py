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

bridge = CvBridge()

def initial_position(i):
    while i<2:
        #print("initial position homnieeee")
        pose = Pose()
        X, Y, Z = 0.0, 0.0, 1.0
        pose.position.x = X
        pose.position.y = Y
        pose.position.z = Z
        pose.orientation.x = 0 # -0.018059104681
        pose.orientation.y = 0 #0.734654724598
        pose.orientation.z = 0 #0.00352329877205
        pose.orientation.w = 1 #0.678191721439
        #x: -0.143055766821
        #y: 0.708422839642
        #z: 0.188000872731
        #w: 0.665077328682

        '''
        x: -0.711743354797
        y: 0.0712868496776
        z: 0.696057736874
        w: 0.0619942210615

        x: 0.05726262182
        y: -0.0194747447968
        z: 1.03383910656
        w: 1.0

        x: -0.018059104681
        y: 0.734654724598
        z: 0.00352329877205
        w: 0.678191721439
        '''
        pub5.publish(pose)
        i+=1
        sleep(4)


rospy.init_node('Face_detection_node', anonymous=True)
pub = rospy.Publisher('/Face_recognition/Searching', String, queue_size=10)
pub2 = rospy.Publisher('/camera_jetson/image_raw', Image, queue_size=100)
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
#initial_position(0)
img_count = 0
rate = rospy.Rate(10) 
flag = False
img_count = 0
while not rospy.is_shutdown():
    #rospy.loginfo("Initial position done")
    #'''
    ret, img = cap.read()
    faces = face_cascade.detectMultiScale(img, 1.3, 9)
    myFaceListC = []
    myFaceListArea = []
    if len(faces) > 0:
        rospy.loginfo("Face detected")
        print("Array faces is:\n", faces)
        #print('Lenght face: ', len(faces))
        #print('Faces type: ', type(faces))
        print('Faces shape: ', faces.shape)
        message_stringg = 'face_found'
        pub4.publish(message_stringg)
        font = cv2.FONT_HERSHEY_DUPLEX
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            c1 =x+(w//2) # Center of BBox X
            c2 =y+(h//2) # Center of BBox Y
            A = h*w  # Area of Bounding box
            roi = img[y:y + h, x:x + w]
            roi = cv2.resize(roi, (96, 96)) 

            '''
            if img_count < 200:
                img_name = "/media/jetson/B2E4-69EA2/Face_recognition_data_analysis/images/test_{}.jpg".format(img_count)
                cv2.imwrite(img_name, roi)
            if img_count < 200:
                with open('/media/jetson/B2E4-69EA2/Face_recognition_data_analysis/min_distance_data.csv', 'a', newline='') as f:
                    write_csv = csv.writer(f)
                    #write_csv.writerow(['col1', 'col2', 'col3'])
                    write_csv.writerow([A, w, h])
                    f.close()
            '''

            print("Xf: ", c1, "Yf: ", c2)
            Bb_size = str(A)
            cv2.circle(img,(c1,c2),10,(0,255,0),2)
            #cv2.putText(img, A, (0,20), font, 1.0, (255, 0, 0), 1)
            coordinates = Point(x=c1, y=A, z=c2)
            pub3.publish(coordinates)
            rospy.loginfo("Bounding Box area: %s", A)                
            img_count += 1
                
            flag = True

            '''
            if c1 > 480:
                rospy.loginfo("go left until get the center")
            if c1 <= 160:
                rospy.loginfo("go to the right until get the center")
            '''

        cv2.putText(img, Bb_size, (0,60), font, 1.0, (255, 0, 0), 1)  
        cv2.putText(img, str(c1), (0,120), font, 1.0, (255, 0, 0), 1)      
        cv2.putText(img, 'Yo homie! A face was found!', (0,20), font, 1.0, (255, 0, 0), 1)

    else:
        rospy.loginfo("Face not found")
        message_string = 'Searching'
        pub.publish(message_string)
        #logging.debug('No face detected: {}'.format(message_string))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    '''
    new_h = frame_height//2
    new_w = frame_width//2
    resize = cv2.resize(gray, (new_w, new_h))
    image_message = bridge.cv2_to_imgmsg(resize)
    pub2.publish(image_message)
    '''
    if ret:
        video_out.write(img)
    
    
    #cv2.imshow('frame', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    rate.sleep()
    

video_out.release()
#cap.release()
#cv2.destroyAllWindows()
