#!/usr/bin/env python
'''
Main node for RPi slave
Subscribe to master RPi and publish to Arduino due.
@DiegoHerrera
'''

import rospy
import ast
from std_msgs.msg import String
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from time import sleep
import re
from tf.transformations import quaternion_from_euler

class Drone_1(object):
    def __init__(self):
        self.sub = rospy.Subscriber("/Test/manual_coordinates", String, self.callback)

    def callback(self, coordinates):
        """
        Callback function for data received from master node. The received data is processed and published to the drone
         node.
         """

        numbers = re.compile('[-+]?\d*\.\d+|\d+')
        list_of_numbers = list(map(float, numbers.findall(str(coordinates))))
        print(list_of_numbers)

        xVal, yVal, zVal, rVal, pVal, yawVal = list_of_numbers[0], list_of_numbers[1], list_of_numbers[2], list_of_numbers[3], list_of_numbers[4], list_of_numbers[5] # modified
        # publish as geometry_msgs/Point msg type
        #coordinates = Point(x=xVal, y=yVal, z=zVal)
        pose = Pose()
        pose.position.x = xVal
        pose.position.y = yVal
        pose.position.z = zVal
        quat = quaternion_from_euler(rVal, pVal, yawVal) # + pi_2)
        print("Quaternion", quat)
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        print("Pose and Orientation: ", pose)  #"X: ", coordinates.x, "Y: ", coordinates.y, "Z: ", coordinates.z)
        pub.publish(pose)
        rate.sleep()


if __name__ == "__main__":
    print("Quadcopter node ready")
    rospy.init_node('Quadcopter_node', anonymous=True)
    #pub = rospy.Publisher('SOKA_DRONE', Pose, queue_size=10)
    pub = rospy.Publisher('/Face_recognition/local_position', Pose, queue_size=10)
    rate = rospy.Rate(10)  # 10hz
    Drone_class = Drone_1()
    rospy.spin()
