#!/usr/bin/env python
'''
This code is to control the position of the drone in the Z axis
Author: @Diego Herrera
email: alberto18_90@outlook.com
'''
import cv2
import rospy
import mavros
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler
from time import sleep

class data_processing():        
    '''
    def orientation_callback(self, msg):
    	self.orientation_value = msg.pose[1].orientation
        self.orientation_value_x = msg.pose[1].orientation.x
        self.orientation_value_y = msg.pose[1].orientation.y
        self.orientation_value_z = msg.pose[1].orientation.z
        self.orientation_value_w = msg.pose[1].orientation.w

    '''
    def orientation_t265_callback(self, msg):
        self.orientation_value = msg.pose.pose
        self.orientation_value_x = msg.pose.pose.orientation.x
        self.orientation_value_y = msg.pose.pose.orientation.y
        self.orientation_value_z = msg.pose.pose.orientation.z
        self.orientation_value_w = msg.pose.pose.orientation.w
    


    def __init__(self):
        self.orientation_value = 0
    	#self.sub = rospy.Subscriber("/gazebo/model_states",ModelStates, self.orientation_callback)
        self.sub = rospy.Subscriber("/camera/odom/sample",Odometry, self.orientation_t265_callback)
        pub = rospy.Publisher('/Logger/orientation', Pose, queue_size=10)
        pu2 = rospy.Publisher('/Logger/position', Pose, queue_size=10)
        self.d = rospy.Duration(0.2)
        while True:
        	rospy.loginfo("Position and Orientation: %s", self.orientation_value)
            #rospy.sleep(self.d)
            


if __name__ == "__main__":
    rospy.init_node('Log_node', anonymous=True)
    rospy.loginfo("node ready")
    rate = rospy.Rate(10)  # 10hz
    log_data = data_processing()
    rospy.spin()