#!/usr/bin/env python

import rospy
from mavros_msgs.srv import SetMode
from mavros_msgs.srv import CommandBool
from mavros_msgs.srv import CommandTOL
import time


# Set Mode
print("\nSetting Mode")
rospy.wait_for_service('/mavros/set_mode')
try:
    change_mode = rospy.ServiceProxy('/mavros/set_mode', SetMode)
    response = change_mode(custom_mode="OFFBOARD")
    rospy.loginfo(response)
except rospy.ServiceException as e:
    print("Set mode failed: %s" %e)


# Arm
print("\nArming")
rospy.wait_for_service('/mavros/cmd/arming')
try:
    arming_cl = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
    response = arming_cl(value = True)
    rospy.loginfo(response)
except rospy.ServiceException as e:
    print("Arming failed: %s" %e)


# Takeoff
print("\nTaking off")
rospy.wait_for_service('/mavros/cmd/takeoff')
try:
    # rosservice call /mavros/cmd/land min_pitch= 0.0, yaw= 0.0, latitude= 0.0, longitude= 0.0, altitude= 0.0
    takeoff_cl = rospy.ServiceProxy('/mavros/cmd/takeoff', CommandTOL)
    response = takeoff_cl(min_pitch= 0.0, yaw= 0.0, latitude= 47.397742, longitude= 8.5455936, altitude= 489.6)
    rospy.loginfo(response)
except rospy.ServiceException, e:
    rospy.loginfo("Service call failed")


