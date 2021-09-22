#!/usr/bin/env python
'''
This code is to control the position of the drone in the Z axis
Author: @Diego Herrera
email: alberto18_90@outlook.com
'''

import rospy
import mavros
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler
from time import sleep

# Callback method for state subscriber
current_state = State()  # Reading the current state from mavros msgs


def state_callback(state):
    global current_state
    current_state = state


mavros.set_namespace()
local_position_publisher = rospy.Publisher(mavros.get_topic('setpoint_position', 'local'), PoseStamped, queue_size=10)  #
local_position_publisher_2 = rospy.Publisher(mavros.get_topic('setpoint_position', 'local'), PoseStamped, queue_size=10) # /mavros/setpoint_position/local
state_subscriber = rospy.Subscriber(mavros.get_topic('state'), State, state_callback)

arming_client = rospy.ServiceProxy(mavros.get_topic('cmd', 'arming'), CommandBool)
set_mode_client = rospy.ServiceProxy(mavros.get_topic('set_mode'), SetMode)

pose = PoseStamped()


def coordinates_orientation(data):
    pose1 = PoseStamped()
    #print("Callback function")
    pose1.pose.position.x = data.position.x
    pose1.pose.position.y = data.position.y
    pose1.pose.position.z = data.position.z
    #quat = quaternion_from_euler(R, P, Y)
    pose1.pose.orientation.x = data.orientation.x
    pose1.pose.orientation.y = data.orientation.y
    pose1.pose.orientation.z = data.orientation.z
    pose1.pose.orientation.w = data.orientation.w
    pose1.header.stamp = rospy.Time.now()
    print(pose1)
    #sleep(5)
    #local_position_publisher.publish(pose1)


def coordinates_orientation_2(data):
    pose2 = PoseStamped()
    #print("Callback function")
    pose2.pose.position.x = data.position.x
    pose2.pose.position.y = data.position.y
    pose2.pose.position.z = data.position.z
    #quat = quaternion_from_euler(R, P, Y)
    pose2.pose.orientation.x = data.orientation.x
    pose2.pose.orientation.y = data.orientation.y
    pose2.pose.orientation.z = data.orientation.z
    pose2.pose.orientation.w = data.orientation.w
    pose2.header.stamp = rospy.Time.now()
    print(pose2)
    #sleep(5)
    #local_position_publisher_2.publish(pose2)
  

def position_control():
    print("Position control function")
    #rospy.init_node('Offboard_node', anonymous=True)
    prev_state = current_state
    rate = rospy.Rate(20.0)

    # Sending a few points before start
    for i in range(100):
        local_position_publisher.publish(pose)
        rate.sleep()

    # We need to wait for FCU connection
    while not current_state.connected:
        rate.sleep()

    last_request = rospy.get_rostime()

    while not rospy.is_shutdown():
        now = rospy.get_rostime()
        
        if current_state.mode != "OFFBOARD" and (now - last_request > rospy.Duration(5.)):
            set_mode_client(base_mode=0, custom_mode="OFFBOARD")
            last_request = now

        else:
            if not current_state.armed and (now - last_request > rospy.Duration(5.)):
                arming_client(True)
                last_request = now

        if current_state.armed:
            print("Drone ready to fly")
        if prev_state.armed != current_state.armed:
            rospy.loginfo("Vehicle armed: %r" % current_state.armed)
        if prev_state.mode != current_state.mode:
            rospy.loginfo("Current mode: %s" % current_state.mode)

        prev_state = current_state
        #print("Hello world")
        #orientation = None
        #velocity = None
        sub = rospy.Subscriber("/SOKA_DRONE/Searching", Pose, coordinates_orientation)
        sub2 = rospy.Subscriber("/SOKA_DRONE/face_found", Pose, coordinates_orientation_2)
        #pose.header.stamp = rospy.Time.now()
        local_position_publisher.publish(pose)
        rate.sleep()


if __name__ == '__main__':
    try:
        rospy.init_node('Offboard_node', anonymous=True)
        position_control()
    except rospy.ROSInterruptException:
        pass
