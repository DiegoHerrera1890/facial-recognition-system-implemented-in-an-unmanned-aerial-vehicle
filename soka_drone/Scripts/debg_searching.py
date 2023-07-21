#!/usr/bin/env python
'''
Searching debugg. It publish the coordinates to rotate the drone according to the keyboard input.
date: October 2021
@DiegoHerrera
'''

import rospy
import ast
import mavros
import math
from std_msgs.msg import String
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler
import numpy as np
from time import sleep
import tty
import sys
import termios

mavros.set_namespace()

class debugg_data():
    
    def search_callback(self, msg):
        self.face_search = True   

    def found_callback(self, msg):
        #print(msg.data)
        self.face_found = True
        return self.face_found

    def match_callback(self, msg):
        self.face_notmatch = True
        return self.face_notmatch

    def kill_callback(self, msg):
        self.kill_program = True

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
        self.orientation_value_y = msg.pose[1].orientation.y
        # self.orientation_value_z = msg.pose[1].orientation.z
        # self.orientation_value_w = msg.pose[1].orientation.w
    # '''

    def __init__(self):
        self.error_z = 0
        self.yawVal = 0.0
        self.count = 1
        self.face_found = False
        self.face_search = False
        self.face_notmatch = False
        self.kill_program = False
        self.des_yawrate = 0.2
        self.c1 = 0 
        self.c2 = 0 
        self.A = 0
        self.orientation_value_y = 0
        self.sleep_time = 0
        self.sleep_time_2 = 0
        self.count_rot_z = 0
        self.count_X = 0

        self.orig_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin)
        self.k = 0
        self.rate = rospy.Rate(10)  # 10hz
        rospy.Subscriber("/Face_recognition/landing/kill_searching", String, self.kill_callback)
        #self.sub = rospy.Subscriber("/gazebo/model_states",ModelStates, self.orientation_callback)
        self.sub = rospy.Subscriber("/camera/odom/sample",Odometry, self.orientation_t265_callback)
        pub = rospy.Publisher('/Face_recognition/coordinates', Pose, queue_size=10)
        pub2 = rospy.Publisher('/Face_recognition/yaw_angle',Float64, queue_size=10)
        ############################# Main part ########################
        self.two_pi = 2 * math.pi
        self.pose = Pose()
        self.d = rospy.Duration(0.3)
        while not rospy.is_shutdown():
        	Y, Z = 0.02, 1.0
	        self.pose.position.y = Y
	        self.pose.position.z = Z
        	k=sys.stdin.read(1)[0]
        	
        	rVal, pVal = 0, 0  
	        if self.kill_program:
	            self.kill_program = False
	            break
        	if k == 'c':
        		print(k)
        		break

        	elif k == 'q':
        		self.yawVal = self.count_rot_z
        		quat = quaternion_from_euler(rVal, pVal, self.yawVal)
        		self.pose.orientation.x = quat[0]
        		self.pose.orientation.y = quat[1]
        		self.pose.orientation.z = quat[2]
        		self.pose.orientation.w = quat[3]
        		pub.publish(self.pose)
        		rospy.loginfo(str(k))
                #print(self.count_rot_z)
        		self.count_rot_z += 0.1
        		print(self.count_rot_z)

        	elif k == 'e':
        		self.yawVal = self.count_rot_z
        		quat = quaternion_from_euler(rVal, pVal, self.yawVal)
        		self.pose.orientation.x = quat[0]
        		self.pose.orientation.y = quat[1]
        		self.pose.orientation.z = quat[2]
        		self.pose.orientation.w = quat[3]
        		pub.publish(self.pose)
        		rospy.loginfo(str(k))
        		self.count_rot_z -= 0.1
        		print(self.count_rot_z)

        	elif k == 'w':
        		self.pose.position.x = self.count_X 
        		pub.publish(self.pose)
        		rospy.loginfo(str(k))
        		self.count_X += 0.03
        		print(self.count_X)
        	elif k == 's':
        		self.pose.position.x = self.count_X 
        		pub.publish(self.pose)
        		rospy.loginfo(str(k))
        		self.count_X -= 0.03
        		print(self.count_X)



	        #rospy.loginfo("Yaw angle: ", self.yawVal)
	        print("Yaw angle: ", self.yawVal)

	        if self.yawVal >= 6.199999999999994:
	            rospy.loginfo("One rotation complete")
	            self.yawVal = 0
	            self.count_X = 0

	        #rospy.loginfo("Pose_data: %s, Twist_data: %s", self.orientation_position, self.angular_velocity)           
        termios.tcgetattr(sys.stdin, termios.TCSADRAIN, self.orig_settings) 
        self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node('debugging_node', anonymous=True)
    rospy.loginfo("Debugging node ready")
    try:
        debugg_data = debugg_data()
    except rospy.ROSInterruptException:
        raise