#!/usr/bin/env python
import getch
import rospy
from std_msgs.msg import String  # String message
from std_msgs.msg import Int8
import sys


################################
# created by yuvaram
# yuvaramsingh94@gmail.com
################################


def keys():
    pub = rospy.Publisher('key', Int8, queue_size=10)  # "key" is the publisher name
    rospy.init_node('keypress', anonymous=True)
    rate = rospy.Rate(10)  # try removing this line ans see what happens
    while not rospy.is_shutdown():
        k = ord(getch.getch())  # this is used to convert the keypress event in the keyboard or joypad , joystick to a
        # ord value
        if k == 27:
            break
        elif k == 224:  # Special keys (arrows, f keys, ins, del, etc.)
            k = ord(getch.getch())
            if k == 80:  # Down arrow
                rospy.loginfo(str(k))  # to print on  terminal
                pub.publish(k)  # to publish
            elif k == 72:  # Up arrow
                rospy.loginfo(str(k))  # to print on  terminal
                pub.publish(k)  # to publish
    rate.sleep()


# s=115,e=101,g=103,b=98

if __name__ == '__main__':
    try:
        keys()
    except rospy.ROSInterruptException:
        pass
