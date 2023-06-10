# ! /usr/bin/env python

import rospy
import mavros
from mavros_msgs.msg import State
from mavros_msgs.msg import ActuatorControl
from sensor_msgs.msg import CameraInfo


def camera_info_callback(msg):
    print("Camera Info: {}".format(msg))


if __name__ == '__main__':
    try:
        rospy.init_node('Camera Info', anonymous=True)
        rospy.Subscriber('/camera/fisheye1/camera_info', CameraInfo, camera_info_callback)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass