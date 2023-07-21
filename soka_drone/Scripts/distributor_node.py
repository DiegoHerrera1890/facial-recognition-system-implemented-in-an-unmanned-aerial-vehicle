#!/usr/bin/env python
'''
Main node for RPi slave
Subscribe to master RPi and publish to Arduino due.
@DiegoHerrera
'''

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import logging
from time import sleep

logging.basicConfig(filename='Distributor_node.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')


class Drone_1():

    def callback(self, data):
        pose = Pose()
        pose.position.x = data.position.x
        pose.position.y = data.position.y
        pose.position.z = data.position.z
        pose.orientation.x = data.orientation.x  # quat[0]
        pose.orientation.y = data.orientation.y  # quat[1]
        pose.orientation.z = data.orientation.z  # quat[2]
        pose.orientation.w = data.orientation.w  # quat[3]
        pub.publish(pose)
        rospy.loginfo("Pose_data: %s, Orientation_data: %s", pose.position, pose.orientation)

    def callback_2(self, data):
        pose = Pose()
        pose.position.x = data.position.x
        pose.position.y = data.position.y
        pose.position.z = data.position.z
        pose.orientation.x = data.orientation.x  # quat[0]
        pose.orientation.y = data.orientation.y  # quat[1]
        pose.orientation.z = data.orientation.z  # quat[2]
        pose.orientation.w = data.orientation.w  # quat[3]
        pub.publish(pose)
        rospy.loginfo("Pose_initial : %s, Orientation_initial: %s", pose.position, pose.orientation)

    def kill_callback(self, data):
        self.kill_program = True

    def __init__(self):
        self.sub = rospy.Subscriber("/Face_recognition/coordinates", Pose, self.callback)
        self.sub2 = rospy.Subscriber("/Face_recognition/initial_position", Pose, self.callback_2)
        # rospy.Subscriber("/Face_recognition/landing/kill_searching", String, self.kill_callback)
        # self.sub3 = rospy.Subscriber("/Test/manual_coordinates", String, self.callback_3)
        self.d = rospy.Duration(0.5)

        while not rospy.is_shutdown():
            landing = raw_input('landing?')
            if landing == 'y':
                rospy.loginfo("Landing detected")
                msg_String = 'landing'
                pose = Pose()
                pose.position.x = 0.08
                pose.position.y = 0.05
                pose.position.z = 0.24
                pose.orientation.x = 0
                pose.orientation.y = 0
                pose.orientation.z = 0
                pose.orientation.w = 1
                pub.publish(pose)
                rospy.loginfo("first position")
                rospy.loginfo("Pose: %s, Orientation: %s", pose.position, pose.orientation)
                sleep(6)
                pose.position.x = -0.09
                pose.position.y = -0.05
                pose.position.z = 0.24
                pose.orientation.x = 0
                pose.orientation.y = 0
                pose.orientation.z = 0
                pose.orientation.w = 1
                pub2.publish(msg_String)
                rospy.loginfo("second position")
                rospy.loginfo("Pose: %s, Orientation: %s", pose.position, pose.orientation)
                sleep(8)              
            rospy.sleep(self.d)


if __name__ == "__main__":
    print("Distributor node ready")
    rospy.init_node('Distributor_node', anonymous=True)
    pub = rospy.Publisher('/Face_recognition/local_position', Pose, queue_size=10)
    pub2 = rospy.Publisher('/Face_recognition/landing', String, queue_size=10)
    rate = rospy.Rate(10)  # 10hz
    Drone_class = Drone_1()
    rospy.spin()