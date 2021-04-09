#!/usr/bin/env python
# license removed for brevity
'''
Simple ROS publisher
'''
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Point


def puto(data):
    print("ruso huevo loco")
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    print("ruso huevo loco")


def talker():
    pub = rospy.Publisher('data_for_ESP8266', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        print("Yendo a callback function")
        rospy.Subscriber("SOKA_DRONE", String, puto)
        pub.publish(hello_str)
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
