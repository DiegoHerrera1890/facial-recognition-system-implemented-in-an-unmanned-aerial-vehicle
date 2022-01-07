#!/usr/bin/env python
'''
Searching node. This node subscribe data from face_recognition node to know
if there is a face or not in front of the camera. If there is no a face then the drone must rotate
over its z axis. This node publish the coordinates to rotate the drone to the distributor node.
date: August 2021
@DiegoHerrera
'''

import rospy
import ast
import mavros
import math
from std_msgs.msg import String
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import PositionTarget  
from mavros_msgs.msg import AttitudeTarget
from gazebo_msgs.msg import ModelStates
from tf.transformations import quaternion_from_euler
import numpy as np
from time import sleep
import logging

mavros.set_namespace()


class data_processing():
    
    def search_callback(self, msg):
        self.face_search = True   

    def found_callback(self, msg):
        #print(msg)
        self.face_found = True
        return self.face_found

    def coordinates(self, msg):
        self.c1 = msg.x # Center of the BBox X
        self.A = msg.y  # Center of the BBox Y
        self.c2 = msg.z # Area of BBox



    def tracking_():
        print('homies')


    def orientation_t265_callback(self, msg):
        self.orientation_position = msg.pose.pose
        self.angular_velocity = msg.twist.twist.angular.z

    # '''
    def orientation_callback(self, msg):
        self.error_x = 0 - (msg.pose[1].position.x)
        self.error_y = 0 - (msg.pose[1].position.y)
        # self.orientation_value_y = msg.pose[1].orientation.y
        # self.orientation_value_z = msg.pose[1].orientation.z
        # self.orientation_value_w = msg.pose[1].orientation.w
    # '''

    def __init__(self):
        self.error_x = 0
        self.error_y = 0
        self.error_z = 0
        self.yawVal = 0.0
        self.count = 1
        self.face_found = False
        self.face_search = False
        self.des_yawrate = 0.2
        self.c1 = 0 
        self.c2 = 0 
        self.A = 0
        self.rate = rospy.Rate(10)  # 10hz
        rospy.Subscriber("/Face_recognition/Searching", String, self.search_callback)
        rospy.Subscriber("/Face_recognition/face_found", String, self.found_callback)
        #rospy.Subscriber("/Face_recognition/face_coordinates", Point, self.coordinates)
        self.sub = rospy.Subscriber("/gazebo/model_states",ModelStates, self.orientation_callback)
        #self.sub = rospy.Subscriber("/camera/odom/sample",Odometry, self.orientation_t265_callback)
        pub = rospy.Publisher('/Face_recognition/coordinates', Pose, queue_size=10)
        pub2 = rospy.Publisher('/Face_recognition/yaw_angle',Float64, queue_size=10)
        ############################# Main part ########################
        self.two_pi = 2 * math.pi
        self.pose = Pose()
        self.d = rospy.Duration(0.3)
        while self.yawVal <= self.two_pi:
            X, Y, Z = 0.05, 0.05, 1.1
            rVal, pVal = 0, 0  
            if self.face_search:
                rospy.loginfo("Looking for faces")
                self.pose.position.x = X
                self.pose.position.y = Y
                self.pose.position.z = Z
                quat = quaternion_from_euler(rVal, pVal, self.yawVal)
                self.pose.orientation.x = quat[0]
                self.pose.orientation.y = quat[1]
                self.pose.orientation.z = quat[2]
                self.pose.orientation.w = quat[3]            
                pub.publish(self.pose)

                self.yawVal += 0.1
                self.face_search = False
            if self.face_found:
                rospy.loginfo("Face located")
                rospy.loginfo("Area: %s", self.A)
                #rospy.loginfo("Pose_data: %s", self.orientation_position)
                pub2.publish(self.yawVal)

                self.face_found = False
            if self.yawVal == 6.199999999999994:
                rospy.loginfo("One rotation complete")
                self.yawVal = 0

            rospy.loginfo("Yaw angle: %s", self.yawVal)
            #rospy.loginfo("Pose_data: %s, Twist_data: %s", self.orientation_position, self.angular_velocity)           
            rospy.sleep(self.d)


if __name__ == "__main__":
    rospy.init_node('Searching_node', anonymous=True)
    rospy.loginfo("Searching node ready")
    try:
        drone_data = data_processing()
    except rospy.ROSInterruptException:
        raise



    '''
    print("Searching node ready")
    rospy.init_node('Searching_node', anonymous=True)
    # pub_vel = rospy.Publisher(mavros.get_topic('setpoint_velocity', 'cmd_vel_unstamped'), Twist, queue_size=10)
    pub = rospy.Publisher('/Face_recognition/coordinates', Pose, queue_size=10)
    pub_raw = rospy.Publisher(mavros.get_topic('setpoint_raw', 'local'), PositionTarget, queue_size=10)
    pub_att = rospy.Publisher(mavros.get_topic('setpoint_attitude', 'cmd_vel'), TwistStamped, queue_size=10)
    #rate = rospy.Rate(10)  # 10hz
    drone_data = data_processing()
    rospy.spin()
    '''