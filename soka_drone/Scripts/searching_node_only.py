#!/usr/bin/env python
'''
Searching node. This node subscribe data from face_recognition node to know
if there is a face or not in front of the camera. If there is no a face then the drone must rotate
over its z axis. This node publish the coordinates to rotate the drone to the distributor node.
date: August 2021
@DiegoHerrera
'''

import rospy
import mavros
import math
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler

from time import sleep


mavros.set_namespace()


class data_processing():

    def kill_callback(self, msg):
        self.kill_program = True


    def __init__(self):
        self.yawVal = 0.0
        self.kill_program = False
        self.rate = rospy.Rate(10)  # 10hz

        rospy.Subscriber("/Face_recognition/landing/kill_searching", String, self.kill_callback)
        pub = rospy.Publisher('/Face_recognition/coordinates', Pose, queue_size=10)
        ############################# Main part ########################
        self.two_pi = 2 * math.pi
        self.pose = Pose()
        self.d = rospy.Duration(0.3)
        while self.yawVal <= self.two_pi:
            X, Y, Z = 0.05, 0.05, 1.01
            rVal, pVal = 0, 0  
            if self.kill_program:
                self.kill_program = False
                break
            rospy.loginfo("Looking for faces")
            quat = quaternion_from_euler(rVal, pVal, self.yawVal)
            self.pose.orientation.x = quat[0]
            self.pose.orientation.y = quat[1]
            self.pose.orientation.z = quat[2]
            self.pose.orientation.w = quat[3]            
            self.pose.position.x = X
            self.pose.position.y = Y
            self.pose.position.z = 1.01

            pub.publish(self.pose)
            self.yawVal += 0.1
            

            if self.yawVal >= 6.199999999999994:
                rospy.loginfo("One rotation complete")
                self.yawVal = 0

            rospy.loginfo("Yaw angle: %s", self.yawVal)
            rospy.sleep(self.d)


if __name__ == "__main__":
    rospy.init_node('Searching_node', anonymous=True)
    rospy.loginfo("Searching node ready")
    try:
        drone_data = data_processing()
    except rospy.ROSInterruptException:
        raise
