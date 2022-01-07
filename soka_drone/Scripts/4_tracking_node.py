#!/usr/bin/env python
'''
Main node for RPi slave
Subscribe to master RPi and publish to Arduino due.
@DiegoHerrera
'''

import rospy
import ast
from std_msgs.msg import String
from std_msgs.msg import Float64
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from time import sleep
import re
from tf.transformations import quaternion_from_euler
import numpy as np
from time import sleep
import math

class data_processing():        

    def angle_callback(self, msg):
        self.yaw_angle = msg.data
        rospy.loginfo("Yaw angle: %s", self.yaw_angle)
        return self.yaw_angle
  

    def callback(self, data):
        self.c1 = data.x
        self.area = data.y
        self.c2 = data.z


    def face_foubd_callback(self, msg):
        found = msg.data
        if found == 'face_found':
            self.flag = True
        if found == 'face_match':
            self.flag2 = True


    def orientation_callback(self, msg):
        self.orientation_value_x = msg.pose[1].orientation.x
        self.orientation_value_y = msg.pose[1].orientation.y
        self.orientation_value_z = msg.pose[1].orientation.z
        self.orientation_value_w = msg.pose[1].orientation.w

    '''
    def orientation_t265_callback(self, msg):
        self.orientation_value_x = msg.pose.pose.orientation.x
        self.orientation_value_y = msg.pose.pose.orientation.y
        self.orientation_value_z = msg.pose.pose.orientation.z
        self.orientation_value_w = msg.pose.pose.orientation.w
    '''

    def backward(self, quadrant, angle):
        if quadrant == 1:
            rospy.loginfo("moving backward %s") 
            X1 = -(self.distance*(math.cos(angle)) + 0.01)
            Y1 = -(self.distance*(math.sin(angle)) + 0.01)
        if quadrant == 2:
            X1 = (self.distance*(math.sin(angle - self.pi_half)) + 0.01)
            Y1 = -(self.distance*(math.cos(angle - self.pi_half)) + 0.01)
        if quadrant == 3:
            X1 = (self.distance*(math.cos(angle - self.pi)) + 0.01)
            Y1 = (self.distance*(math.sin(angle - self.pi)) + 0.01)
        if quadrant == 4:
            X1 = -(self.distance*(math.sin(angle - self.pi_three_half)) + 0.01)
            Y1 = (self.distance*(math.cos(angle - self.pi_three_half)) + 0.01)
        rospy.loginfo("Quadruant: %s", quadrant)
        rospy.loginfo("X coordinate is: %s", X1)
        rospy.loginfo("Y coordinate is: %s", Y1)
        Z = self.altitude
        self.pose.position.x = X1
        self.pose.position.y = Y1
        self.pose.position.z = Z
        self.pose.orientation.x = self.orientation_value_x
        self.pose.orientation.y = self.orientation_value_y
        self.pose.orientation.z = self.orientation_value_z
        self.pose.orientation.w = self.orientation_value_w  
        


    def forward(self, quadrant, angle):
        if quadrant == 1:
            rospy.loginfo("moving forward") 
            X1 = (self.distance*(math.cos(angle)) + 0.01)
            Y1 = (self.distance*(math.sin(angle)) + 0.01)
        if quadrant == 2:
            X1 = -(self.distance*(math.sin(angle - self.pi_half)) + 0.01)
            Y1 = (self.distance*(math.cos(angle - self.pi_half)) + 0.01)
        if quadrant == 3:
            X1 = -(self.distance*(math.cos(angle - self.pi)) + 0.01)
            Y1 = -(self.distance*(math.sin(angle - self.pi)) + 0.01)
        if quadrant == 4:
            X1 = (self.distance*(math.sin(angle - self.pi_three_half)) + 0.01)
            Y1 = -(self.distance*(math.cos(angle - self.pi_three_half)) + 0.01)
        rospy.loginfo("Quadruant: %s", quadrant)
        rospy.loginfo("X coordinate is: %s", X1)
        rospy.loginfo("Y coordinate is: %s", Y1)
        Z = self.altitude
        self.pose.position.x = X1
        self.pose.position.y = Y1
        self.pose.position.z = Z
        self.pose.orientation.x = self.orientation_value_x
        self.pose.orientation.y = self.orientation_value_y
        self.pose.orientation.z = self.orientation_value_z
        self.pose.orientation.w = self.orientation_value_w


    def rigth():
        rospy.loginfo("Go rigth")
        


    def left():
        rospy.loginfo("go left")


    #def odometry_callback(self, msg):
    #    self.odom_data = msg.pose.pose.orientation


    def __init__(self):
        self.pi_half = math.pi/2
        self.pi = math.pi
        self.pi_three_half = (3*math.pi)/2
        self.two_pi = 2*math.pi
        self.distance = 0.0
        self.altitude = 0.68
        self.flag = False
        self.flag2 = False
        self.c1 = 0 
        self.c2 = 0 
        self.area = 0
        self.yaw_angle = 0
        self.odom_data = 0
        self.first_quad = 1
        self.second_quad = 2
        self.third_quad = 3
        self.fourth_quad = 4
        self.pose = Pose()
        self.pose.orientation.x = 0
        self.pose.orientation.y = 0
        self.pose.orientation.z = 0 
        self.pose.orientation.w = 0
        self.orientation_value_x = 0 
        self.orientation_value_y = 0 
        self.orientation_value_z = 0 
        self.orientation_value_w = 0 
        self.sub = rospy.Subscriber("/Face_recognition/face_found", String, self.face_foubd_callback)
        self.sub = rospy.Subscriber("/Face_recognition/yaw_angle", Float64, self.angle_callback)
        self.sub = rospy.Subscriber("/Face_recognition/face_coordinates", Point, self.callback)
        self.sub = rospy.Subscriber("/gazebo/model_states",ModelStates, self.orientation_callback)
        #self.sub = rospy.Subscriber("/camera/odom/sample",Odometry, self.orientation_t265_callback)
        pub = rospy.Publisher('/Face_recognition/coordinates', Pose, queue_size=10)
        self.d = rospy.Duration(0.3)
        while True:
            if self.flag:
                #print('area: ', self.area)
                if self.area >= 22000:
                    rospy.loginfo("Face too close get away, please %s", self.area)
                    rospy.loginfo("Yaw angle: %s", self.yaw_angle)
                    rospy.loginfo("Distance to move: %s", self.distance)
                    if 0<= self.yaw_angle <= self.pi_half:
                        self.backward(self.first_quad, self.yaw_angle)
                    if self.pi_half< self.yaw_angle <= self.pi:
                        self.backward(self.second_quad, self.yaw_angle)
                    if self.pi< self.yaw_angle <= self.pi_three_half:
                        self.backward(self.third_quad, self.yaw_angle)
                    if self.pi_three_half< self.yaw_angle <= self.two_pi:
                        self.backward(self.fourth_quad, self.yaw_angle)
                    self.distance += 0.02
                    pub.publish(self.pose)
                    self.flag = False
                if self.area <= 4000:
                    rospy.loginfo("Face too far get closer, please %s", self.area)
                    rospy.loginfo("Yaw angle: %s", self.yaw_angle)
                    rospy.loginfo("Distance to move: %s", self.distance)
                    if 0<= self.yaw_angle <= self.pi_half:
                        self.forward(self.first_quad, self.yaw_angle)
                    if self.pi_half< self.yaw_angle <= self.pi:
                        self.forward(self.second_quad, self.yaw_angle)
                    if self.pi< self.yaw_angle <= self.pi_three_half:
                        self.forward(self.third_quad, self.yaw_angle)
                    if self.pi_three_half< self.yaw_angle <= self.two_pi:
                        self.forward(self.fourth_quad, self.yaw_angle)
                    self.distance += 0.02
                    pub.publish(self.pose)
                    self.flag = False
                if 4000<self.area<2200:
                    rospy.loginfo("Safety area of %s and holding position", self.area)
                    if self.flag2:
                        if self.c1 > 460:
                            self.right()
                        if self.c1 <= 180:
                            self.left()
                    else:
                        pass
                        #hold its position

            #if self.flag:
            #    print('puto')
            rospy.sleep(self.d)
            


if __name__ == "__main__":
    rospy.init_node('Tracking_node', anonymous=True)
    rospy.loginfo("Tracking node ready")
    pub = rospy.Publisher('/Face_recognition/coordinates', Pose, queue_size=10)
    rate = rospy.Rate(10)  # 10hz
    drone_data = data_processing()
    rospy.spin()