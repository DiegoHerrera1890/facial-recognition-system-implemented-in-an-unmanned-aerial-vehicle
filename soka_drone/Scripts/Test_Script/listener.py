#!/usr/bin/env python
'''
Simple ROS subscriber
'''

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Point

def callback(data):
    #rospy.loginfo(rospy.get_caller_id() + "I heard %s",Point)
    X = data.x
    Y = data.y
    Z = data.z
    print("X value: ", X)
    print("Y value: ", Y)
    print("Z value: ", Z)
    print("")
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("SOKA_DRONE6", Point, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
