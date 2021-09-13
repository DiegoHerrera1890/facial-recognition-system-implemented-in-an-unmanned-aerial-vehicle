#!/usr/bin/env python
'''
Main node for RPi slave
Subscribe to master RPi and publish to Arduino due.
@DiegoHerrera
'''

import rospy
import ast
import mavros
from std_msgs.msg import String
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseStamped
from time import sleep
import re
from tf.transformations import quaternion_from_euler
import numpy as np
from time import sleep
mavros.set_namespace()

class data_processing():
    def __init__(self):
        self.yawVal = 0
        self.sub = rospy.Subscriber("/Face_recognition/Searching", String, self.callback) 
             
    
    def callback(self, data):
        two_pi = 2*3.14159265359
        pose = Pose()
        #velocity_msg = Twist()
        if data.data =="Searching"and self.yawVal<two_pi:
            
            X = 0
            Y = 0
            Z = 1.6
            rVal = 0
            pVal = 0 
            self.yawVal += 0.1
            print("Yaw value: ", self.yawVal) 
            pose.position.x = X
            pose.position.y = Y
            pose.position.z = Z
            quat = quaternion_from_euler(rVal, pVal, self.yawVal)
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            #print("position: ", pose.orientation)
            pub.publish(pose)
            
            
        else:
            self.yawVal = 0 



if __name__ == "__main__":
    print("Searching node ready")
    rospy.init_node('Searching_node', anonymous=True)
    #setpoint_velocity_publisher = rospy.Publisher(mavros.get_topic('setpoint_velocity', 'cmd_vel_unstamped'), Twist, queue_size=10)
    pub = rospy.Publisher('/Face_recognition/coordinates', Pose, queue_size=10)
    rate = rospy.Rate(10)  # 10hz
    drone_data = data_processing()
    rospy.spin()
