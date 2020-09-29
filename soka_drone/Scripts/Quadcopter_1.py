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
from time import sleep
import re


class Drone_1(object):
    def __init__(self):
        self.sub = rospy.Subscriber("Quadcopter_coordinates", String, self.callback)

    def callback(self, coordinates):
        """
        Callback function for data received from master node. The received data is processed and published to the drone
         node.
         """

        numbers = re.compile('[-+]?\d*\.\d+|\d+')
        list_of_numbers = list(map(float, numbers.findall(str(coordinates))))
        print(list_of_numbers)

        xVal, yVal, zVal = list_of_numbers[0], list_of_numbers[1], list_of_numbers[2]
        # publish as geometry_msgs/Point msg type
        coordinates = Point(x=xVal, y=yVal, z=zVal)
        print("X: ", coordinates.x, "Y: ", coordinates.y, "Z: ", coordinates.z)
        pub.publish(coordinates)
        rate.sleep()


if __name__ == "__main__":
    print("Quadcopter node ready")
    rospy.init_node('Quadcopter_node', anonymous=True)
    pub = rospy.Publisher('SOKA_DRONE', Point, queue_size=10)
    rate = rospy.Rate(10)  # 10hz
    Drone_class = Drone_1()
    rospy.spin()
