#!/usr/bin/env python
"""
This code is to control the position of the drone in the Z axis
Author: @Diego Herrera
email: alberto18_90@outlook.com
"""
import rospy
import mavros
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
# from gazebo_msgs.msg import ModelStates
from time import sleep

# current_state = State()  # Reading the current state from mavros msgs

mavros.set_namespace()


class drone_control:

    def state_callback(self, state):
        self.current_state = state

    def coordinates_orientation_2(self, data_2):
        # print('data of Face Found: ', data_2)
        self.pose.header.stamp = rospy.Time.now()
        self.pose.pose.position.x = data_2.position.x
        self.pose.pose.position.y = data_2.position.y
        self.pose.pose.position.z = data_2.position.z
        # quat = quaternion_from_euler(rVal2, pVal2, yawVal2) # + pi_2)
        self.pose.pose.orientation.x = data_2.orientation.x  # quat[0]
        self.pose.pose.orientation.y = data_2.orientation.y  # quat[1]
        self.pose.pose.orientation.z = data_2.orientation.z  # quat[2]
        self.pose.pose.orientation.w = data_2.orientation.w  # quat[3]

    def land_callback(self, msg):
        self.flag = True

    def __init__(self):
        self.current_state = State()
        self.flag = False
        self.pose = PoseStamped()
        self.local_position_publisher = rospy.Publisher(mavros.get_topic('setpoint_position', 'local'), PoseStamped,
                                                        queue_size=10)  #
        self.sub = rospy.Subscriber("/Face_recognition/local_position", Pose, self.coordinates_orientation_2)
        self.state_subscriber = rospy.Subscriber(mavros.get_topic('state'), State, self.state_callback)
        self.arming_client = rospy.ServiceProxy(mavros.get_topic('cmd', 'arming'), CommandBool)
        self.takeoff_client = rospy.ServiceProxy(mavros.get_topic('cmd', 'takeoff'), CommandTOL)
        # landing_client = rospy.ServiceProxy(mavros.get_topic('cmd', 'land'), CommandTOL)
        self.set_mode_client = rospy.ServiceProxy(mavros.get_topic('set_mode'), SetMode)

        print("Position control function")
        self.prev_state = self.current_state
        rate = rospy.Rate(20.0)

        for i in range(100):
            self.local_position_publisher.publish(self.pose)
            rate.sleep()

        while not self.current_state.connected:
            rate.sleep()

        self.last_request = rospy.get_rostime()

        while not rospy.is_shutdown():
            now = rospy.get_rostime()
            # qxVal, yVal, zVal, rVal, pVal, yawVal = 0 , 0, 1.6, 0, 0, 0
            if self.current_state.mode != "OFFBOARD" and (now - self.last_request > rospy.Duration(5.)):
                self.set_mode_client(base_mode=0, custom_mode="OFFBOARD")
                last_request = now

            else:
                if not self.current_state.armed and (now - self.last_request > rospy.Duration(5.)):
                    self.arming_client(True)
                    last_request = now

            if self.current_state.armed:
                print("Drone ready to fly")
            if self.prev_state.armed != self.current_state.armed:
                rospy.loginfo("Vehicle armed: %r" % self.current_state.armed)
            if self.prev_state.mode != self.current_state.mode:
                rospy.loginfo("Current mode: %s" % self.current_state.mode)

            self.prev_state = self.current_state

            # rospy.Subscriber("/Face_recognition/drone_search", Pose, coordinates_orientation)
            self.pose.header.stamp = rospy.Time.now()
            self.local_position_publisher.publish(self.pose)
            rospy.Subscriber("/Face_recognition/landing", String, self.land_callback)
            if self.flag:
                print("hello landing")
                self.set_mode_client(base_mode=0, custom_mode="AUTO.LAND")
                sleep(8)
                self.flag = False
                break

            rate.sleep()


if __name__ == '__main__':
    try:
        rospy.init_node('Quadcopter_node', anonymous=True)
        drone_control()
    except rospy.ROSInterruptException:
        pass