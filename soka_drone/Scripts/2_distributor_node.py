#!/usr/bin/env python
'''
Main node for RPi slave
Subscribe to master RPi and publish to Arduino due.
@DiegoHerrera
'''

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose


class Drone:
    def __init__(self):
        self.sub = rospy.Subscriber("/Face_recognition/coordinates", Pose, self.callback)
        self.sub2 = rospy.Subscriber("/Face_recognition/initial_position", Pose, self.callback_2)
        self.d = rospy.Duration(0.5)

    def callback(self, data):
        pose = Pose()
        pose.position.x = data.position.x
        pose.position.y = data.position.y
        pose.position.z = data.position.z
        pose.orientation.x = data.orientation.x
        pose.orientation.y = data.orientation.y
        pose.orientation.z = data.orientation.z
        pose.orientation.w = data.orientation.w
        pub.publish(pose)
        rospy.loginfo("Pose_data: %s, Orientation_data: %s", pose.position, pose.orientation)

    def callback_2(self, data):
        pose = Pose()
        pose.position.x = data.position.x
        pose.position.y = data.position.y
        pose.position.z = data.position.z
        pose.orientation.x = data.orientation.x
        pose.orientation.y = data.orientation.y
        pose.orientation.z = data.orientation.z
        pose.orientation.w = data.orientation.w
        pub.publish(pose)
        rospy.loginfo("Pose_initial: %s, Orientation_initial: %s", pose.position, pose.orientation)

    def run(self):
        while not rospy.is_shutdown():
            rospy.sleep(self.d)


if __name__ == "__main__":
    print("Distributor node ready")
    rospy.init_node('Distributor_node', anonymous=True)
    pub = rospy.Publisher('/Face_recognition/local_position', Pose, queue_size=10)
    Drone().run()
