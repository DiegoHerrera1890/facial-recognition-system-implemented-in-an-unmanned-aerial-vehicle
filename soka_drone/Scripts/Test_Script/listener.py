#!/usr/bin/env python
'''
Simple ROS subscriber
'''

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler

def coordinates_orientation(data):
    #rospy.loginfo(rospy.get_caller_id() + "I heard %s",Point)
    print("Data: ", data)
    X = data.position.x
    Y = data.position.y
    Z = data.position.z
    Xq = data.orientation.x
    Yq = data.orientation.y
    Zq = data.orientation.z
    Wq = data.orientation.w
    print("X value: ", X)
    print("Y value: ", Y)
    print("Z value: ", Z)
    print("X quaternion: ", Xq)
    print("Y quaternion: ", Yq)
    print("Z quaternion: ", Zq)
    print("W quaternion: ", Wq)
    print("")
def position_control():
    print("Listener ready")
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.

    #rospy.Subscriber("/nalgas/putaso", Point, callback)
    rospy.Subscriber("SOKA_DRONE", Pose, coordinates_orientation)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    try:
        rospy.init_node('Offboard_node', anonymous=True)
        position_control()
    except rospy.ROSInterruptException:
        pass
