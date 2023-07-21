#!/usr/bin/env python3
'''

@DiegoHerrera
'''

import rospy
import mavros
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped


mavros.set_namespace()


def zed_convertion(msg):
	pose = PoseStamped()
	pose_cov = PoseWithCovarianceStamped()
	# Pose with covariance
	pose_cov.header = msg.header
	pose_cov.pose = msg.pose  

	# Pose without covariance
	pose.header = msg.header
	pose.pose = msg.pose.pose

	rospy.loginfo(pose)
	pose_pub.publish(pose)
	pose_cov_pub.publish(pose_cov)

		

def main_converter():
	rospy.init_node('zed_pose_converter', anonymous=True)
	rospy.Subscriber("/zedm/zed_node/odom", Odometry, zed_convertion)
	#rospy.Subscriber("/zedm/zed_node/odom",Odometry, orientation_t265_callback)
	
	rate = rospy.Rate(10)  # 10hz
	while not rospy.is_shutdown():		
		rate.sleep()


if __name__ == '__main__':
    try:
    	pose_pub = rospy.Publisher('/mavros/vision_pose/pose', PoseStamped, queue_size=10)
    	pose_cov_pub = rospy.Publisher('/mavros/vision_pose/pose_cov', PoseWithCovarianceStamped, queue_size=10)
    	main_converter()
    except rospy.ROSInterruptException:
        pass