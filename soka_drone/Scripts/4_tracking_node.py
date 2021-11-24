#!/usr/bin/env python
'''
Main node for RPi slave
Subscribe to master RPi and publish to Arduino due.
@DiegoHerrera
'''

import rospy
import ast
from std_msgs.msg import String
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from time import sleep
import re
from tf.transformations import quaternion_from_euler
import numpy as np
from time import sleep

class data_processing():
    def __init__(self):
        self.X = 0
        self.sub = rospy.Subscriber("/Face_recognition/face_found", Point, self.callback) 
        #self.sub = rospy.Subscriber("/Face_recognition/face_coordinates", Point, self.callback_2)

    def callback(self, data):
        pose = Pose()
        c1 = data.x
        c2 = data.z
        Area = data.y
        print("Area: ", Area)
        if Area >= 17300:
            print("Face too close \n get away, please \n", Area)

        #'''
        if Area <= 15000 and self.X<200:
            print("Face too far \n get closer, please \n", Area)
            #if data.data =="Searching" and self.yawVal<two_pi:
            self.X += 0.1
            print("X value: ", self.X)   
            pose.position.x = self.X
            pose.position.y = 0
            pose.position.z = 1.6
            #print("position: ", pose.orientation)
            pub.publish(pose)
            sleep(.1)      
        else:
            self.X = 0
        #''' 



    def callback2(self, data):
        pose = Pose()
        pose.position.x = data.x
        pose.position.y = data.y
        pose.position.z = data.z
        pose.orientation.x = 0 #quat[0]
        pose.orientation.y = 0 #quat[1]
        pose.orientation.z = 0 #quat[2]
        pose.orientation.w = 1 #quat[3]
        print("position: ", pose.position.z)
        pub.publish(pose)


if __name__ == "__main__":
    print("Tracking node ready")
    rospy.init_node('Tracking_node', anonymous=True)
    pub = rospy.Publisher('/Face_recognition/coordinates', Pose, queue_size=10)
    rate = rospy.Rate(10)  # 10hz
    drone_data = data_processing()
    rospy.spin()