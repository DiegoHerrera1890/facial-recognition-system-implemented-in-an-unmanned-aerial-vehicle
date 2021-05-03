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
from std_msgs.msg import String
from time import sleep

# Callback method for state subscriber
current_state = State()  # Reading the current state from mavros msgs
offb_set_mode = SetMode  # Reading the setmode and saving at off_set_mode


def state_callback(state):
    global current_state
    current_state = state


mavros.set_namespace()
state_subscriber = rospy.Subscriber(mavros.get_topic('state'), State, state_callback)

arming_client = rospy.ServiceProxy(mavros.get_topic('cmd', 'arming'), CommandBool)
landing_client = rospy.ServiceProxy(mavros.get_topic('cmd', 'land'), CommandBool)
set_mode_client = rospy.ServiceProxy(mavros.get_topic('set_mode'), SetMode)

pose = PoseStamped()


def arming_drone():
    global prev_state
    prev_state = current_state
    print("arming drone")
    arming_client(True)
    """
    while not current_state.armed:
        print("The vehicle is not armed, please arm the vehicle manually")
    """
    if current_state.armed:
        print("El drone esta listo pa volar homie")
    if prev_state.armed != current_state.armed:
        rospy.loginfo("Vehicle armed: %r" % current_state.armed)
    if prev_state.mode != current_state.mode:
        rospy.loginfo("Current mode: %s" % current_state.mode)
    prev_state = current_state


def position_control(position_to_go):
    global prev_state
    prev_state = current_state
    print("Position control")
    print("Local_position_publisher____ publishing coordinates")
    local_position_publisher.publish(position_to_go)
    """
    if position_to_go.pose.position.x == 0.0 and position_to_go.pose.position.y == 0.0 and position_to_go.pose.position.z == 0.0:
        print("Los valores son zerosss")
        print("X value is: ", position_to_go.pose.position.x)
        print("Y value is: ", position_to_go.pose.position.y)
        print("Z value is: ", position_to_go.pose.position.z)
        set_mode_client(base_mode=0, custom_mode="MANUAL")
        sleep(2)
    else:
        print("Estamos ready bro!!!!")
        print("X value is: ", position_to_go.pose.position.x)
        print("Y value is: ", position_to_go.pose.position.y)
        print("Z value is: ", position_to_go.pose.position.z)
        sleep(1)
        local_position_publisher.publish(position_to_go)
    """



def callback(data):
    print("Callback function")

    X = data.x
    Y = data.y
    Z = data.z
    print("X value: ", X)
    print("Y value: ", Y)
    print("Z value: ", Z)
    print("")
    # pose = PoseStamped()
    pose.pose.position.x = X
    pose.pose.position.y = Y
    pose.pose.position.z = Z
    # pose.header.stamp = rospy.Time.now()
    # local_position_publisher.publish(pose)


def main_function():
    global prev_state
    prev_state = current_state
    print("Main function")
    # Sending a few points before start
    """
    for i in range(100):
        local_position_publisher.publish(pose)
        rate.sleep()
    """

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
                arming_drone()
                last_request = now
        print("Going to callback function")
        rospy.Subscriber("SOKA_DRONE", Point, callback)
        print("after callback function")
        pose.header.stamp = rospy.Time.now()
        position_control(pose)
        rate.sleep()


if __name__ == '__main__':

    rospy.init_node('Offboard_node', anonymous=True)
    local_position_publisher = rospy.Publisher(mavros.get_topic('setpoint_position', 'local'), PoseStamped,
                                               queue_size=10)
    global prev_state
    rate = rospy.Rate(20.0)
    try:
        main_function()
    except rospy.ROSInterruptException:
        pass
