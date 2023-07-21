#!/usr/bin/env python
'''

@DiegoHerrera
'''

import rospy
from std_msgs.msg import String, Float64
from geometry_msgs.msg import Pose
from time import sleep
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion


class drone_1:

    def takeoff_task(self, data):
        self.takeoff = data.data
        if self.takeoff == 'ready':
            self.takeoff_flag = True
        if self.takeoff == 'done':
            self.takeoff_flag = False
        return self.takeoff_flag

    def orientation_zed_callback(self, msg):
        self.pose_pos_x = msg.pose.pose.position.x
        self.pose_pos_y = msg.pose.pose.position.y
        self.pose_pos_z = msg.pose.pose.position.z

    def orientation_t265_callback(self, msg):
        self.pose_pos_x = msg.pose.pose.position.x
        self.pose_pos_y = msg.pose.pose.position.y
        self.pose_pos_z = msg.pose.pose.position.z
        self.orientation_ = msg.pose.pose.orientation
        quaternion = [self.orientation_.x, self.orientation_.y, self.orientation_.z, self.orientation_.w]
        euler_angles = euler_from_quaternion(quaternion)


    def orientation_callback(self, msg):
        self.pose_pos_x = msg.pose[1].position.x
        self.pose_pos_y = msg.pose[1].position.y
        self.pose_pos_z = msg.pose[1].position.z

    def callback(self, data):
        pose = Pose()
        pose.position.x = data.position.x
        pose.position.y = data.position.y
        pose.position.z = data.position.z
        pose.orientation.x = data.orientation.x  # quat[0]
        pose.orientation.y = data.orientation.y  # quat[1]
        pose.orientation.z = data.orientation.z  # quat[2]
        pose.orientation.w = data.orientation.w  # quat[3]
        self.pub_s.publish(pose)
        rospy.loginfo("Pose_data: %s, Orientation_data: %s", pose.position, pose.orientation)




    def __init__(self):
        rospy.Subscriber("/Face_recognition/model_ready", String, self.takeoff_task)
        rospy.Subscriber("/Face_recognition/coordinates", Pose, self.callback)
        rospy.Subscriber("/camera/odom/sample", Odometry, self.orientation_t265_callback)
        #rospy.Subscriber("/gazebo/model_states",ModelStates, self.orientation_callback)
        pub = rospy.Publisher('/Face_recognition/local_position', Pose, queue_size=10)
        self.pub_s = rospy.Publisher('/Face_recognition/local_position', Pose, queue_size=10)
        landing_pub = rospy.Publisher('/Face_recognition/landing/land', String, queue_size=10)
        pub = rospy.Publisher('/Face_recognition/local_position', Pose, queue_size=10)
        kill_pub = rospy.Publisher('/Face_recognition/landing/kill_searching', String, queue_size=10)

        self.takeoff = ''
        self.takeoff_flag = False
        self.landing_flag = ''
        self.orientation_ = Pose()
        self.yaw_angle = 0
        self.initial_position = 0.14
        self.zVal = 0.70
        self.pose_pos_x = 0
        self.pose_pos_y = 0
        self.pose_pos_z = 0
        self.position_value_z = 0
        self.rate = rospy.Rate(10)  # 10hz
        self.d = rospy.Duration(0.5)
        self.pose = Pose()
        rate = rospy.Rate(20)

        self.count = 0
        self.count2 = 0

        while not rospy.is_shutdown():
            if self.takeoff_flag:
                rospy.loginfo("Takingoff...")
                self.pose.position.x = 0.0
                self.pose.position.y = 0.0
                self.pose.position.z = 0.95
                self.pose.orientation.x = 0  # -0.018059104681
                self.pose.orientation.y = 0  # 0.734654724598
                self.pose.orientation.z = 0  # 0.00352329877205
                self.pose.orientation.w = 1  # 0.678191721439
                pub.publish(self.pose)
                rospy.loginfo("altitude equal to: %f" % self.pose_pos_z)
                

                if self.pose_pos_z >= 0.9:
                    self.landing_flag = raw_input('landing?')
                    pub.publish(self.pose)
                    self.takeoff_flag = False

            
            if self.landing_flag == 'y':
                rospy.loginfo("Landing detected")
                msg_String = 'landing'
                kill_pub.publish(msg_String)
                sleep(15)
                print("done 20 secs")
                pose = Pose()
                rospy.loginfo("first position")
                pose.orientation.x = 0
                pose.orientation.y = 0
                pose.orientation.z = 0
                pose.orientation.w = 1

                # Gradually decrease the z position from 0.80 to 0.10
                for z_position in [0.80, 0.50, 0.38, 0.24, 0.15, 0.10, 0.06]:
                    pose.position.x = 0.02
                    pose.position.y = 0.02
                    pose.position.z = z_position
                    pub.publish(pose)
                    rospy.loginfo("Pose: %s, Orientation: %s", pose.position, pose.orientation)
                    sleep(3)

                rospy.loginfo("Landing complete")
                landing_pub.publish(msg_String)
                sleep(3)
                break            
                    
            if self.pose_pos_z < -0.08:
                rospy.loginfo("Negative altitude, landing")
                msg_String = 'landing'
                landing_pub.publish(msg_String)
                rospy.loginfo("Landing detected")
                sleep(1)
                break

            rospy.sleep(self.d)

            rate.sleep()


if __name__ == "__main__":
    rospy.init_node('takingoff_node', anonymous=True)
    rospy.loginfo("Takeoff_landing node ready")
    try:
        drone_class = drone_1()
    except rospy.ROSInterruptException:
        raise
