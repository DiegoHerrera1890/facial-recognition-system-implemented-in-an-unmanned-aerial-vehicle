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
import logging

logging.basicConfig(filename='Distributor_node.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')



class Drone_1():
    
    def callback(self, data):
        pose = Pose()
        pose.position.x = data.position.x
        pose.position.y = data.position.y
        pose.position.z = data.position.z
        pose.orientation.x = data.orientation.x #quat[0]
        pose.orientation.y = data.orientation.y #quat[1]
        pose.orientation.z = data.orientation.z #quat[2]
        pose.orientation.w = data.orientation.w #quat[3]
        pub.publish(pose) 
        rospy.loginfo("Pose_data: %s, Orientation_data: %s", pose.position, pose.orientation)

    def callback_2(self, data):
        pose = Pose()
        pose.position.x = data.position.x
        pose.position.y = data.position.y
        pose.position.z = data.position.z
        pose.orientation.x = data.orientation.x #quat[0]
        pose.orientation.y = data.orientation.y #quat[1]
        pose.orientation.z = data.orientation.z #quat[2]
        pose.orientation.w = data.orientation.w #quat[3]
        pub.publish(pose)
        rospy.loginfo("Pose_initial: %s, Orientation_initial: %s", pose.position, pose.orientation)

    def __init__(self):
        self.sub = rospy.Subscriber("/Face_recognition/coordinates", Pose, self.callback)
        self.sub2 = rospy.Subscriber("/Face_recognition/initial_position", Pose, self.callback_2)       
        
        while True:
            landing = raw_input('landing?')
            if landing == 'y':
                msg_String = 'landing'
                pub2.publish(msg_String)

if __name__ == "__main__":
    print("Distributor node ready")
    rospy.init_node('Distributor_node', anonymous=True)
    pub = rospy.Publisher('/Face_recognition/local_position', Pose, queue_size=10)
    pub2 = rospy.Publisher('/Face_recognition/landing', String, queue_size=10)
    rate = rospy.Rate(10)  # 10hz
    Drone_class = Drone_1()
    rospy.spin()











