#!/usr/bin/env python
'''
Main node for RPi slave
Subscribe to master RPi and publish to Arduino due.
@DiegoHerrera
'''

import rospy
import ast
import mavros
import math
from std_msgs.msg import String
# from mavros_msgs.msg import Thrust
# from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from mavros_msgs.msg import PositionTarget  # TwistStamped
from mavros_msgs.msg import AttitudeTarget
# from geometry_msgs.msg import PoseStamped
# from time import sleep
# import re
from tf.transformations import quaternion_from_euler
import numpy as np
from time import sleep

mavros.set_namespace()


class data_processing():
    def __init__(self):
        self.yawVal = 0.0
        self.count = 1
        # self.count2 = 1
        self.des_yawrate = 0.2
        # self.att_msg = PositionTarget()
        # self.pos_msg = PoseStamped()
        # self.thr_msg = Thrust()
        self.rate = rospy.Rate(10)  # 10hz

        self.sub = rospy.Subscriber("/Face_recognition/Searching", String, self.rotate_callback)
        # self.velocity_publisher = rospy.Publisher(mavros.get_topic('setpoint_velocity', 'cmd_vel_unstamped'), Twist,
        #                                          queue_size=10)
        # self.pub_att = rospy.Publisher(mavros.get_topic('setpoint_raw', 'local'), PositionTarget, queue_size=100)
        # self.pub_vel = rospy.Publisher(mavros.get_topic('setpoint_raw', 'cmd_vel'), TwistStamped, queue_size=100)
        # self.pub_thr = rospy.Publisher(mavros.get_topic('setpoint_raw', 'thrust'), Thrust, queue_size=10)

    def rotate_callback(self, msg):
        rospy.loginfo("Searching new faces")
        two_pi = 2 * math.pi
        att_msg = PositionTarget()
        while True:
            print("before if statement")
            if self.count == 10000:
                print("inside IF", self.count)
                break
            else:
                pass
            self.count += 1
            print("inside IF", self.count)
            att_msg.header.stamp = rospy.Time.now()
            att_msg.header.seq = 0
            att_msg.header.frame_id = ""
            att_msg.type_mask = 1475
            att_msg.coordinate_frame = 8
            att_msg.position.x = 0.0
            att_msg.position.y = 0.0
            att_msg.position.z = 1.6
            print("at the middle")
            att_msg.velocity.x = 0.0
            att_msg.velocity.y = 0.0
            att_msg.velocity.z = 0.0
            att_msg.acceleration_or_force.x = 0.0
            att_msg.acceleration_or_force.y = 0.0
            att_msg.acceleration_or_force.z = 0.0
            att_msg.yaw = float('nan')
            att_msg.yaw_rate = 0.4
            # self.pub_thr.publish(self.thr_msg)
            print("Publishing attitude msg: \n", att_msg)
            pub_att.publish(att_msg)

            # self.yawVal += 0.1
            # self.pub_vel.publish(self.vel_msg)
        # else:
        #     print("hello")

    def callback2(self, msg):
        rospy.loginfo("Searching new faces")
        two_pi = 2 * math.pi
        att_msg = AttitudeTarget()
        # if msg.data == "Searching" and self.yawVal < two_pi:
        rVal = 0.0
        pVal = 0.0
        yawVal = math.pi
        # self.yawVal += 0.1
        att_msg.header.stamp = rospy.Time.now()
        att_msg.header.seq = 0
        att_msg.header.frame_id = ""
        att_msg.type_mask = 67
        # att_msg.coordinate_frame = 8
        quat = quaternion_from_euler(rVal, pVal, yawVal)
        print(quat)
        att_msg.orientation.x = 0  # quat[0]
        att_msg.orientation.y = 0  # quat[1]
        att_msg.orientation.z = 0.70710678  # quat[2]
        att_msg.orientation.w = 0.70710678  # quat[3]
        #att_msg.thrust = 0.3
        att_msg.body_rate.x = 0.0
        att_msg.body_rate.y = 0.0
        att_msg.body_rate.z = 0.2
        # self.pub_thr.publish(self.thr_msg)
        pub_att.publish(att_msg)
        print('yawVal: ', self.yawVal)
        # rospy.loginfo("yawVal", self.yawVal)
        # self.yawVal += 0.1
        # self.yawVal += 0.1
        # self.pub_vel.publish(self.vel_msg)
        # else:
        #     self.yawVal = 0.0

    def callback(self, data):
        rospy.loginfo("Searching new faces")
        two_pi = 2 * math.pi
        pose = Pose()
        # vel_msg = Twist()
        if data.data == "Searching" and self.yawVal < two_pi:
            X = 0
            Y = 0
            Z = 1.6
            rVal = 0
            pVal = 0
            # vel_msg.angular.z = 0.2
            # pub_vel.publish(vel_msg)
            # print("Vel_msg: ", vel_msg)
            pose.position.x = X
            pose.position.y = Y
            pose.position.z = Z
            quat = quaternion_from_euler(rVal, pVal, self.yawVal)
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            # print("position: ", pose.orientation)
            pub.publish(pose)
            print('yawVal: ', self.yawVal)
            # rospy.loginfo("yawVal", self.yawVal)
            self.yawVal += 0.1

        else:
            self.yawVal = 0


if __name__ == "__main__":
    print("Searching node ready")
    rospy.init_node('Searching_node', anonymous=True)
    # pub_vel = rospy.Publisher(mavros.get_topic('setpoint_velocity', 'cmd_vel_unstamped'), Twist, queue_size=10)
    pub = rospy.Publisher('/Face_recognition/coordinates', Pose, queue_size=10)
    pub_att = rospy.Publisher(mavros.get_topic('setpoint_raw', 'local'), PositionTarget, queue_size=100)
    # pub_att = rospy.Publisher(mavros.get_topic('setpoint_raw', 'attitude'), AttitudeTarget, queue_size=100)
    rate = rospy.Rate(10)  # 10hz
    drone_data = data_processing()
    rospy.spin()
