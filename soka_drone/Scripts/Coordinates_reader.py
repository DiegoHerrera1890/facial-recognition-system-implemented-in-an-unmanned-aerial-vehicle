#!/usr/bin/env python
"""
Created on Fri Jul  3 23:31:20 2020
Publisher and Subscriber
Thi is node is for sending data to the drone robot
according to the XYZ coordinates
@author: Diego Herrera
"""

import rospy
from std_msgs.msg import String
from time import sleep
from command_separator import Command_Parameters


def soka_drone(string_to_send):
    pub.publish(str(string_to_send))
    rospy.loginfo('Soka_drone: %s', string_to_send)
    sleep(1)
    rate.sleep()


def reading_csv():
    """
    Publishes commands contained in Command_Code_Parameters to slave Raspberry Pi
    Command_Code_Parameters format = [Haive ID, Hardware key, Command ID, (command relevant parameters)]
    """

    for code_parameter in Command_Parameters:
        # check HAIVE ID is 4001 or 4002  (HAIVE ID will not be included in the data packet that gets published)
        soka_drone(code_parameter[0:])  # (Hardware key, Command ID, (command relevant parameters))


if __name__ == '__main__':
    rospy.init_node('Main_node', anonymous=True)
    pub = rospy.Publisher('/Test/manual_coordinates', String, queue_size=10)
    sleep(1)
    print("Main node ready")
    rate = rospy.Rate(10)
    try:
        reading_csv()
        print("Done")
    except rospy.ROSInterruptException:
        pass
