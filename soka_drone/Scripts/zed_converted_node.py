#!/usr/bin/env python
"""
Main node for RPi slave
Subscribe to master RPi and publish to Arduino due.
@DiegoHerrera
"""

import rospy
import mavros
from nav_msgs.msg import Odometry
from mavros_msgs.msg import CompanionProcessStatus
import threading


class Drone_1():

    def odom_callback(self, msg):
        output = msg
        output.header.frame_id = msg.header.frame_id
        output.child_frame_id = msg.child_frame_id
        output.pose.pose.position = msg.pose.pose.position
        output.pose.pose.orientation = msg.pose.pose.orientation
        output.twist.covariance = msg.twist.covariance
        output.twist.twist.linear = msg.twist.twist.linear
        output.twist.twist.angular = msg.twist.twist.angular
        
        mavros_odom_pub.publish(output)
        self.flag_first_pose_received = True

        with self.status_lock:
            last_system_status = self.system_status

            if msg.pose.covariance[0] > 0.1:
                self.system_status = 2  # MAV_STATE_FLIGHT_TERMINATION
            elif msg.pose.covariance[0] == 0.1:
                self.system_status = 1  # MAV_STATE_CRITICAL
            elif msg.pose.covariance[0] == 0.01:
                self.system_status = 0  # MAV_STATE_ACTIVE
            else:
                rospy.logwarn("Unexpected vision sensor variance")

            if last_system_status != self.system_status:
                status_msg = CompanionProcessStatus()
                status_msg.header.stamp = rospy.Time.now()
                status_msg.component = 197  # MAV_COMP_ID_VISUAL_INERTIAL_ODOMETRY
                status_msg.state = self.system_status
                mavros_system_status_pub.publish(status_msg)

            self.last_callback_time = rospy.Time.now()

    def publish_system_status(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            if self.flag_first_pose_received:
                if rospy.Time.now() - self.last_callback_time > rospy.Duration(0.5):
                    rospy.logwarn("Stopped receiving data from Zed mini")
                    self.system_status = 1
                status_msg = CompanionProcessStatus()
                status_msg.header.stamp = rospy.Time.now()
                status_msg.component = 197
                status_msg.state = self.system_status
                mavros_system_status_pub.publish(status_msg)
            rate.sleep()

    def __init__(self):
        # self.nh = rospy.init_node('PX4_realsense_bridge')
        #self.odom_sub = rospy.Subscriber("/camera/odom/sample_throttled", Odometry, self.odom_callback)
        self.odom_sub = rospy.Subscriber("/zedm/zed_node/odom", Odometry, self.odom_callback)
        self.flag_first_pose_received = False
        self.last_callback_time = rospy.Time.now()
        self.system_status = 0
        self.last_system_status = 0
        self.status_lock = threading.Lock()
        self.worker = threading.Thread(target=self.publish_system_status)
        self.worker.start()


if __name__ == "__main__":
    rospy.init_node('PX4_Real_sense_Bridge', anonymous=True)
    print("PX4_Real_sense_Bridge node ready")
    mavros_odom_pub = rospy.Publisher("/mavros/odometry/out", Odometry, queue_size=10)
    mavros_system_status_pub = rospy.Publisher("/mavros/companion_process/status", CompanionProcessStatus, queue_size=1)
    rate = rospy.Rate(10)  # 10hz
    Drone_class = Drone_1()
    rospy.spin()