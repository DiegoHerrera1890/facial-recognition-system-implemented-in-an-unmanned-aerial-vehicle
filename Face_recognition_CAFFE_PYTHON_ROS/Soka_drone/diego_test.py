#!/usr/bin/env python
'''
This code is to control the position of the drone in the Z axis

Author: @Diego Herrera
email: alberto18_90@outlook.com
'''

import rospy
import tf
import mavros
from geometry_msgs.msg import Pose, PoseStamped, Twist, Quaternion
from mavros_msgs.msg import OverrideRCIn
from mavros_msgs.msg import RCIn
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool
from mavros_msgs.srv import SetMode
from mavros_msgs.srv import CommandTOL
from geometry_msgs.msg import Point


class mavros_main_control():
    def __init__(self):
        print("Mavros Control")
        mavros.set_namespace()
        rospy.init_node('Offboard_node', anonymous=True)
        print("node already created")
        # Subscribers
        rospy.Subscriber("/mavros/state", State, self.state_callback)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_callback)
        rospy.Subscriber("/mavros/rc/in", RCIn, self.rc_callback)
        rospy.Subscriber("SOKA_DRONE", Point, self.coordinates)
        # Publisher
        self.local_position_publisher = rospy.Publisher(mavros.get_topic('setpoint_position', 'local'), PoseStamped,queue_size=10)
        self.velocity_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=10)
        self.rc_override = rospy.Publisher("/mavros/rc/override", OverrideRCIn, queue_size=10)

        # Mavros services for arming, takeoff, and set mode
        self.arming_service = rospy.ServiceProxy(mavros.get_topic('cmd', 'arming'), CommandBool)
        self.takeoff_service = rospy.ServiceProxy(mavros.get_topic('cmd', 'takeoff'), CommandTOL)
        self.set_mode_service = rospy.ServiceProxy("/mavros/set_mode", SetMode)

        # Flight modes for PX4
        # STABILIZE
        # OFFBOARD
        # AUTO.LAND

        # Callback method for state subscriber
        self.current_state = State()  # Reading the current state from mavros msgs
        self.timestamp = rospy.Time()
        self.pose = Pose()
        self.rc = RCIn()
	self.state = State()
        print("end of the class")
        # while not self.current_state.connected:
        #     print("waiting FCU connection")
        #     rate.sleep()

    def rc_callback(self, data):
        print("rc_callback")
        self.rc = data

    def pose_callback(self, data):
        # print("pose_callback")
        self.timestamp = data.header.stamp
        self.pose = data.pose

    def state_callback(self, data):
        # print("state_callback")
        self.current_state = data

    def arm(self):
        """
        return: vehicle is armed and ready to fly
        """
        return self.arming_service(True)

    def disarm(self):
        """
        return: vehicle is disarmed
        """
        return self.arming_service(False)

    def control_mode(self):
        self.set_mode_service(base_mode=0, custom_mode="OFFBOARD")
        # print("Offboard Mode")
        # print("Current mode: %s" % self.current_state.mode)
        # return current_mode

    def takeoff(self):  # height=2.0
        """
        :param height: Desire altitude to achieve
        :return current_mode: the current mode of the quadrotor
        """
        current_mode = self.set_mode_service(base_mode=0, custom_mode="OFFBOARD")
	rospy.loginfo("Current mode: %s" % self.state.mode)
        self.arm()

        # Takingoff the drone
        # min_pitch= 0.0, yaw= 0.0, latitude= 47.397742, longitude= 8.5455936, altitude= 489.6
        self.takeoff_service()  # (altitude=489.height)

        # return current_mode

    def land(self):
        resp = self.set_mode_service(base_mode=0, custom_mode="AUTO.LAND")
        self.disarm()

    def coordinates(self, data):
        print("Callback function")

        X = data.x
        Y = data.y
        Z = data.z
        print("X value: ", X)
        print("Y value: ", Y)
        print("Z value: ", Z)
        print("")
        pose = Pose()
        pose.position.x = X
        pose.position.y = Y
        pose.position.z = Z
        self.goto(pose)

    def goto(self, pose):
        """
        :param pose: Set the target as the next setpoint by sending Local_Position_NED
        """
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.timestamp
        pose_stamped.pose = pose

        self.local_position_publisher.publish(pose_stamped)


def position_control():
    print("Position control")
    control = mavros_main_control()
    rate = rospy.Rate(10)  # 10hz
    while not control.current_state.connected:
        print("waiting FCU connection")
        rate.sleep()
    control.control_mode()
    last_request = rospy.get_rostime()
    control.takeoff()
    print("before while")

    while not rospy.is_shutdown():
        now = rospy.get_rostime()
        #control.takeoff()
        #print("before while")

        if control.current_state.armed:
            print("El drone esta listo pa volar homie")
        # if prev_state.armed != current_state.armed:
        #     rospy.loginfo("Vehicle armed: %r" % current_state.armed)
        # if prev_state.mode != current_state.mode:
        #     rospy.loginfo("Current mode: %s" % current_state.mode)

        # rospy.Subscriber("SOKA_DRONE", Point, control.coordinates())
        # print("after callback function")
        # pose.header.stamp = rospy.Time.now()
        # print("Local_position_publisher____ publishing coordinates")
        # local_position_publisher.publish(pose)
        rate.sleep()


if __name__ == '__main__':
    try:
        position_control()
    except rospy.ROSInterruptException:
        pass
