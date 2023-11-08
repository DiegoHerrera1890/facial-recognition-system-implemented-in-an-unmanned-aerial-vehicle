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
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import numpy as np
from time import sleep
import logging
import math

mavros.set_namespace()


class data_processing():
    
    def search_callback(self, msg):
        search = msg.data
        if search == 'Searching':
            self.face_search = True
            self.face_search2 = False

        if search == 'Unknown':
            self.face_search = False
            self.face_search2 = True



    def found_callback(self, msg):
        if msg.data == 'face_found':
            self.face_found = True
        else:
            self.face_found = False
        return self.face_found

    
    def coordinates(self, msg):
        self.c1 = msg.x # Center of the BBox X
        self.A = msg.y  # Center of the BBox Y
        self.c2 = msg.z # Area of BBox


    def tracking_():
        print('homies')


    def orientation_t265_callback(self, msg):
        self.orientation_ = msg.pose.pose.orientation
        quaternion = [self.orientation_.orientation.x, self.orientation_.orientation.y, self.orientation_.orientation.z, self.orientation_.orientation.w]
        #print("euler angles: ", euler_angles)

    
    def orientation_callback(self, msg):
        self.orientation_.orientation = msg.pose[1].orientation
        quaternion = [self.orientation_.orientation.x, self.orientation_.orientation.y, self.orientation_.orientation.z, self.orientation_.orientation.w]
        euler_angles = euler_from_quaternion(quaternion)
        #print("euler angles: ", euler_angles)
    

    def kill_callback(self, msg):
        if msg.data == 'landing':
            self.kill_program = True


    def yaw_fb_callback(self, msg):
        self.yawVal = msg.data


    def __init__(self):
        self.error_x = 0
        self.error_y = 0
        self.error_z = 0
        self.pi = math.pi
        self.pi_half = math.pi/2
        self.two_pi = 2*math.pi
        self.yawVal = 0.0
        self.yawVal_n = 0
        self.count = 1
        self.face_found = False
        self.face_search = False
        self.face_search2 = False
        #self.face_notmatch = False
        self.kill_program = False
        self.des_yawrate = 0.2
        self.c1 = 0 
        self.c2 = 0 
        self.A = 0
        self.orientation_value_y = 0
        self.sleep_time = 0
        self.sleep_time2 = 0
        self.orientation_ = Pose()
        #self.sleep_time_2 = 0
        self.rate = rospy.Rate(10)  # 10hz
        rospy.Subscriber("/Face_recognition/Searching", String, self.search_callback)
        rospy.Subscriber("/Face_recognition/face_found", String, self.found_callback)
        rospy.Subscriber("/Face_recognition/landing/kill_searching", String, self.kill_callback)
        rospy.Subscriber("/Face_recognition/yaw_angle_fb", Float64, self.yaw_fb_callback)
        '''
        rospy.Subscriber("/Face_recognition/face_notmatch", String, self.match_callback)
        self.sub = rospy.Subscriber("/gazebo/model_states",ModelStates, self.orientation_callback)
        self.sub = rospy.Subscriber("/camera/odom/sample",Odometry, self.orientation_t265_callback)
        rospy.Subscriber("/Face_recognition/face_coordinates", Point, self.coordinates)
        '''
        self.sub = rospy.Subscriber("/gazebo/model_states",ModelStates, self.orientation_callback)
        pub = rospy.Publisher('/Face_recognition/coordinates', Pose, queue_size=10)
        pub2 = rospy.Publisher('/Face_recognition/yaw_angle',Float64, queue_size=10)
        pub3 = rospy.Publisher('/Face_recognition/yaw_angle_2',Float64, queue_size=10)
        ############################# Main part ########################
        
        self.pose = Pose()
        self.pose2 = Pose()
        self.orientation_ = Pose()
        self.d = rospy.Duration(0.3)
        while self.yawVal <= self.two_pi:
            X, Y, Z = 0.05, 0.05, 1.01
            rVal, pVal = 0, 0  
            if self.kill_program:
                if 0 <= self.yawVal <= self.pi:
                    while self.yawVal >= 0.15:
                        quat = quaternion_from_euler(rVal, pVal, self.yawVal)
                        self.pose.orientation.x = quat[0]
                        self.pose.orientation.y = quat[1]
                        self.pose.orientation.z = quat[2]
                        self.pose.orientation.w = quat[3]            
                        self.pose.position.x = X
                        self.pose.position.y = Y
                        self.pose.position.z = 1.01
                        pub.publish(self.pose)
                        self.yawVal = self.yawVal- 0.1
                        rospy.loginfo("Yaw angle: %s", self.yawVal)
                        rospy.sleep(self.d)
                    

                if self.pi <= self.yawVal <= self.two_pi:
                    while self.yawVal <= 6:
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
                        rospy.loginfo("Yaw angle: %s", self.yawVal)
                        rospy.sleep(self.d)

                break

            if self.face_search and not self.kill_program and not self.face_found:
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
                pub2.publish(self.yawVal)
                self.yawVal += 0.1
                self.sleep_time2 = 0
                self.face_search = False

            if self.face_found and not self.face_search2 and not self.kill_program:
                rospy.loginfo("Face located")
                rospy.loginfo("Area: %s", self.A)
                # sleep for a while
                if self.sleep_time < 5:
                    rospy.sleep(.7)
                    self.sleep_time += 1
                if self.sleep_time2 < 2:
                    pub2.publish(self.yawVal)
                    self.sleep_time2 +=1
                self.face_found = False
           
            if self.face_search2: 
                rospy.loginfo("Face not matching")
                #print('yawVal notmatch: ', self.yawVal)
        
                self.pose.position.x = X
                self.pose.position.y = Y
                self.pose.position.z = Z
                self.yawVal += 0.3
                quat = quaternion_from_euler(rVal, pVal, self.yawVal)
                self.pose.orientation.x = quat[0]
                self.pose.orientation.y = quat[1]
                self.pose.orientation.z = quat[2]
                self.pose.orientation.w = quat[3]            
                pub.publish(self.pose)
                self.sleep_time2 = 0
                self.face_search2 = False
                
            
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
