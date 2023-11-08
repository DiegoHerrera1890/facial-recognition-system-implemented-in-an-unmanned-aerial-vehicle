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
#import gpio
from tf.transformations import quaternion_from_euler
#from vision_msgs.msg import Detection2DArray
import numpy as np
from time import sleep
import math

class data_processing():        

    def angle_callback(self, msg):
        self.yaw_angle = msg.data
        
        

    def object_detection(self, data):
        self.bbox = data.detections[0].bbox
        self.bbox_x = data.detections[0].bbox.center.x
        self.bbox_y = data.detections[0].bbox.center.y
        if self.bbox_x > 0:
            self.flag3 = True
        else:
            self.flag3 = False
  

    def coordinate_callback(self, data):
        self.c1 = data.x # Center of Bbox X
        self.area = data.y # area of Bbox
        #self.c2 = data.z # Center of Bbox Y
        #rospy.loginfo("C1 is %f", self.c1)


    def face_found_callback(self, msg):
        found = msg.data
        if found == 'face_found':
            self.flag = True
            
        if found == 'Unknown':
            self.flag = False


    def face_match_callback(self, msg):
        match = msg.data 
        if match == 'face_match':
            self.flag2 = True
        if match == 'Unknown':
            self.flag2 = False
    
    def orientation_callback(self, msg):
        self.orientation_value_x = msg.pose[1].orientation.x
        self.orientation_value_y = msg.pose[1].orientation.y
        self.orientation_value_z = msg.pose[1].orientation.z
        self.orientation_value_w = msg.pose[1].orientation.w
    

    def kill_callback(self, msg):
        self.kill_program = True

    def orientation_t265_callback(self, msg):
        self.pose_orientation_x = msg.pose.pose.position.x
        self.pose_orientation_y = msg.pose.pose.position.y
        self.pose_orientation_z = msg.pose.pose.position.z
        self.orientation_value_x = msg.pose.pose.orientation.x
        self.orientation_value_y = msg.pose.pose.orientation.y
        self.orientation_value_z = msg.pose.pose.orientation.z
        self.orientation_value_w = msg.pose.pose.orientation.w
    
    def right(self, angle_n):
        rospy.loginfo("Go rigth")
        rospy.loginfo("Yaw angle is %f", angle_n)
        rVal, pVal = 0, 0
        quat = quaternion_from_euler(rVal, pVal, angle_n)
        self.pose.position.x = self.pose.position.x
        self.pose.position.y = self.pose.position.y
        self.pose.position.z = self.pose.position.z
        self.pose.orientation.x = quat[0]
        self.pose.orientation.y = quat[1]
        self.pose.orientation.z = quat[2]
        self.pose.orientation.w = quat[3] 
        #self.pub.publish(self.pose)

    def left(self, angle_n):
        rospy.loginfo("go left")
        rVal, pVal = 0, 0
        quat = quaternion_from_euler(rVal, pVal, angle_n)
        self.pose.position.x = self.pose.position.x
        self.pose.position.y = self.pose.position.y
        self.pose.position.z = self.pose.position.z
        self.pose.orientation.x = quat[0]
        self.pose.orientation.y = quat[1]
        self.pose.orientation.z = quat[2]
        self.pose.orientation.w = quat[3]
        

    def hold_position(self, angle_n):
        rVal, pVal = 0, 0
        quat = quaternion_from_euler(rVal, pVal, angle_n)
        self.pose.position.x = self.pose_orientation_x
        self.pose.position.y = self.pose_orientation_y
        self.pose.position.z = self.pose_orientation_z
        self.pose.orientation.x = quat[0]
        self.pose.orientation.y = quat[1]
        self.pose.orientation.z = quat[2]
        self.pose.orientation.w = quat[3]


    def new_quaternion(self, data):        
        
        self.orientation_x = data.orientation.x #quat[0]
        self.orientation_y = data.orientation.y #quat[1]
        self.orientation_z = data.orientation.z #quat[2]
        self.orientation_w = data.orientation.w #quat[3] 


    def backward(self, quadrant, angle):
        quat = quaternion_from_euler(0, 0, angle)
        rospy.loginfo("Quaternion: %s", quat)
        if quadrant == 1:
            rospy.loginfo("moving backward ") 
            X1 = -(self.distance*(math.cos(angle)) + 0.09)
            Y1 = -(self.distance*(math.sin(angle)) + 0.09)
        if quadrant == 2:
            X1 = (self.distance*(math.sin(angle - self.pi_half)) + 0.09)
            Y1 = -(self.distance*(math.cos(angle - self.pi_half)) + 0.09)
        if quadrant == 3:
            X1 = (self.distance*(math.cos(angle - self.pi)) + 0.09)
            Y1 = (self.distance*(math.sin(angle - self.pi)) + 0.09)
        if quadrant == 4:
            X1 = -(self.distance*(math.sin(angle - self.pi_three_half)) + 0.09)
            Y1 = (self.distance*(math.cos(angle - self.pi_three_half)) + 0.09)
        rospy.loginfo("Quadruant: %s", quadrant)
        rospy.loginfo("X coordinate is: %s", X1)
        rospy.loginfo("Y coordinate is: %s", Y1)
        Z = self.altitude
        self.pose.position.x = X1
        self.pose.position.y = Y1
        self.pose.position.z = Z
        self.pose.orientation.x = self.orientation_x 
        self.pose.orientation.y = self.orientation_y 
        self.pose.orientation.z = self.orientation_z 
        self.pose.orientation.w = self.orientation_w 
        rospy.loginfo("orientation X is: %s", self.pose.orientation.x) 
        rospy.loginfo("orientation Y is: %s", self.pose.orientation.y) 
        rospy.loginfo("orientation Z is: %s", self.pose.orientation.z) 
        rospy.loginfo("orientation W is: %s", self.pose.orientation.w) 

        
    def forward(self, quadrant, angle):
        quat = quaternion_from_euler(0, 0, angle)
        rospy.loginfo("Quaternion: %s", quat)
        if quadrant == 1:
            rospy.loginfo("moving forward") 
            X1 = (self.distance*(math.cos(angle)) + 0.09)
            Y1 = (self.distance*(math.sin(angle)) + 0.09)
        if quadrant == 2:
            X1 = -(self.distance*(math.sin(angle - self.pi_half)) + 0.09)
            Y1 = (self.distance*(math.cos(angle - self.pi_half)) + 0.09)
        if quadrant == 3:
            X1 = -(self.distance*(math.cos(angle - self.pi)) + 0.09)
            Y1 = -(self.distance*(math.sin(angle - self.pi)) + 0.09)
        if quadrant == 4:
            X1 = (self.distance*(math.sin(angle - self.pi_three_half)) + 0.09)
            Y1 = -(self.distance*(math.cos(angle - self.pi_three_half)) + 0.09)
        rospy.loginfo("Quadruant: %s", quadrant)
        rospy.loginfo("X coordinate is: %s", X1)
        rospy.loginfo("Y coordinate is: %s", Y1)
        Z = self.altitude
        self.pose.position.x = X1
        self.pose.position.y = Y1
        self.pose.position.z = Z
        self.pose.orientation.x = self.orientation_x 
        self.pose.orientation.y = self.orientation_y 
        self.pose.orientation.z = self.orientation_z 
        self.pose.orientation.w = self.orientation_w 
        rospy.loginfo("orientation X is: %s", self.pose.orientation.x) 
        rospy.loginfo("orientation Y is: %s", self.pose.orientation.y) 
        rospy.loginfo("orientation Z is: %s", self.pose.orientation.z) 
        rospy.loginfo("orientation W is: %s", self.pose.orientation.w) 


   



    def __init__(self):
        self.pi = math.pi
        self.pi_half = math.pi/2
        self.pi_three_half = (3*math.pi)/2
        self.two_pi = 2*math.pi
        self.distance = 0.1
        self.altitude = 1.0
        self.flag = False
        self.flag2 = False
        self.flag3 = False
        self.kill_program = False
        self.c1 = 0 
        #self.c2 = 0 
        self.area = 0
        self.yaw_angle = 0
        self.first_quad = 1
        self.second_quad = 2
        self.third_quad = 3
        self.fourth_quad = 4
        self.pose = Pose()
        self.pose.position.x = 0
        self.pose.position.y = 0
        self.pose.position.z = 1.15
        self.pose.orientation.x = 0
        self.pose.orientation.y = 0
        self.pose.orientation.z = 0 
        self.pose.orientation.w = 1
        self.bbox_x = 0   
        self.count = 0 
        self.orientation_x = 0 #quat[0]
        self.orientation_y = 0 #quat[1]
        self.orientation_z = 0 #quat[2]
        self.orientation_w = 1 #quat[3] 
        self.sub = rospy.Subscriber("/Face_recognition/face_found", String, self.face_found_callback)
        self.sub = rospy.Subscriber("/Face_recognition/face_coordinates", Point, self.coordinate_callback)
        self.sub = rospy.Subscriber("/Face_recognition/yaw_angle", Float64, self.angle_callback)
        self.sub = rospy.Subscriber("/Face_recognition/coordinates", Pose, self.new_quaternion)
        self.sub = rospy.Subscriber("/Face_recognition/landing/kill_searching", String, self.kill_callback)
        self.pub = rospy.Publisher('/Face_recognition/coordinates', Pose, queue_size=10)
        self.pub_yaw = rospy.Publisher('/Face_recognition/yaw_angle_fb', Float64, queue_size=10)

        self.d = rospy.Duration(0.1)
        while not rospy.is_shutdown():
            #rospy.loginfo("Yaw angle is: %s", self.yaw_angle)
            if self.kill_program:
                self.kill_program = False
                break
            if self.flag: # and self.sleep_time >= 5
                #rospy.loginfo("Face was found")
                rospy.sleep(.4)
                #print('area: ', self.area)
                if self.area >= 78501:
                    rospy.loginfo("Face too close get away, please %s", self.area)
                    rospy.loginfo("Yaw angle: %s", self.yaw_angle)
                    rospy.loginfo("Distance to move: %s", self.distance)
                    """
                    if 0<= self.yaw_angle <= self.pi_half:
                        self.backward(self.first_quad, self.yaw_angle)
                    if self.pi_half< self.yaw_angle <= self.pi:
                        self.backward(self.second_quad, self.yaw_angle)
                    if self.pi< self.yaw_angle <= self.pi_three_half:
                        self.backward(self.third_quad, self.yaw_angle)
                    if self.pi_three_half< self.yaw_angle <= self.two_pi:
                        self.backward(self.fourth_quad, self.yaw_angle)
                    self.distance += 0.02
                    self.pub.publish(self.pose)
                    """
                    self.flag = False
                if self.area <= 5700:
                    rospy.loginfo("Face too far get closer, please %s", self.area)
                    rospy.loginfo("Yaw angle: %s", self.yaw_angle)
                    rospy.loginfo("Distance to move: %s", self.distance)
                    """
                    if 0<= self.yaw_angle <= self.pi_half:
                        self.forward(self.first_quad, self.yaw_angle)
                    if self.pi_half< self.yaw_angle <= self.pi:
                        self.forward(self.second_quad, self.yaw_angle)
                    if self.pi< self.yaw_angle <= self.pi_three_half:
                        self.forward(self.third_quad, self.yaw_angle)
                    if self.pi_three_half< self.yaw_angle <= self.two_pi:
                        self.forward(self.fourth_quad, self.yaw_angle)
                    self.distance += 0.02
                    self.pub.publish(self.pose)
                    """
                    self.flag = False
                    
                if 5701<self.area<78500:
                    rospy.loginfo("Safety area of %s and holding position", self.area)

                    if self.flag: #and self.flag3:    # self.flag2
                        #rospy.loginfo("C1 is: %f, and yaw angle is: %f" %(self.c1, self.yaw_angle))
                        if self.c1 <= 180:
                            self.yaw_angle += 0.1
                            self.right(self.yaw_angle)
                            #rospy.loginfo("Yaw angle: %s", self.yaw_angle)
                            self.pub.publish(self.pose)
                            self.pub_yaw.publish(self.yaw_angle)
                        if self.c1 >= 460:
                            self.yaw_angle -= 0.1
                            self.left(self.yaw_angle)
                            #rospy.loginfo("Yaw angle: %s", self.yaw_angle)
                            self.pub.publish(self.pose)
                            self.pub_yaw.publish(self.yaw_angle)
                        if 300<self.c1<900:
                            rospy.loginfo(" Face in the center with yaw angle: %s", self.yaw_angle)
                        rospy.loginfo(" yaw_angle: %f", self.yaw_angle)
                        self.flag = False
                    

                    else:
                        pass
                        #hold its position
            #rospy.sleep(self.d)
            


if __name__ == "__main__":
    rospy.init_node('Tracking_node', anonymous=True)
    rospy.loginfo("Tracking node ready")
    rate = rospy.Rate(10)  # 10hz
    drone_data = data_processing()
    rospy.spin()