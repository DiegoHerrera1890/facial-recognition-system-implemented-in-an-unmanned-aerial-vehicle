#!/usr/bin/env python
'''
Main node for RPi slave
Subscribe to master RPi and publish to Arduino due.
@DiegoHerrera
'''

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from time import sleep


class DroneController:
    def __init__(self):
        self.takeoff_flag = False
        self.landing_flag = False
        self.pose_pos_z = 0

        rospy.Subscriber("/Face_recognition/model_ready", String, self.takeoff_task)
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.orientation_callback)
        rospy.Subscriber("/camera/odom/sample", Odometry, self.orientation_t265_callback)
        self.pub = rospy.Publisher('/Face_recognition/local_position', Pose, queue_size=10)
        self.landing_pub = rospy.Publisher('/Face_recognition/landing', String, queue_size=10)

    def takeoff_task(self, data):
        self.takeoff_flag = data.data == 'ready'

    def orientation_callback(self, msg):
        if len(msg.pose) > 1:
            pose_1 = msg.pose[1]
            self.pose_pos_z = pose_1.position.z

    def orientation_t265_callback(self, msg):
        if len(msg.pose) > 1:
            pose_1 = msg.pose
            self.pose_pos_x = pose_1.pose.position.x
            self.pose_pos_y = pose_1.pose.position.y
            self.pose_pos_z = pose_1.pose.position.z

    def takeoff(self):
        rospy.loginfo("Taking off...")
        self.send_pose(0.0, 0.0, 0.75, 0, 0, 0, 1)

        while not rospy.is_shutdown():
            if self.pose_pos_z >= 0.7:
                self.landing_flag = raw_input('Landing? (y/n): ')
                self.send_pose(0.0, 0.0, 0.75, 0, 0, 0, 1)
                self.takeoff_flag = False

            if self.landing_flag.lower() == 'y':
                self.landing_sequence()
                break

            rospy.sleep(0.5)

    def send_pose(self, x, y, z, qx, qy, qz, qw):
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        self.pub.publish(pose)

    def landing_sequence(self):
        rospy.loginfo("Landing detected")
        msg_string = 'landing'
        positions = [
            (0.08, 0.05, 0.24),
            (-0.09, -0.05, 0.24)
        ]

        for position in positions:
            self.send_pose(position[0], position[1], position[2], 0, 0, 0, 1)
            rospy.loginfo("Pose: %s, Orientation: %s", position[:3], (0, 0, 0, 1))
            sleep(6)

        self.landing_pub.publish(msg_string)
        sleep(5)

    def run(self):
        rospy.init_node('takingoff_node', anonymous=True)
        rospy.loginfo("Takeoff_landing node ready")

        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            if self.takeoff_flag:
                self.takeoff()

            if self.pose_pos_z < -0.05:
                rospy.loginfo("Landing detected")
                msg_string = 'landing'
                self.landing_pub.publish(msg_string)

            rate.sleep()


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

    drone_controller = DroneController()
    drone_controller.run()

    Drone().run()
