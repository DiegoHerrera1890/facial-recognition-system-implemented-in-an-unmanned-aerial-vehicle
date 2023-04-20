#!/usr/bin/env python
'''

@DiegoHerrera
'''

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from time import sleep
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates


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

    def orientation_callback(self, msg):
        self.pose_pos_x = msg.pose[1].position.x
        self.pose_pos_y = msg.pose[1].position.y
        self.pose_pos_z = msg.pose[1].position.z


    def __init__(self):
        rospy.Subscriber("/Face_recognition/model_ready", String, self.takeoff_task)
        rospy.Subscriber("/camera/odom/sample", Odometry, self.orientation_t265_callback)
        #rospy.Subscriber("/zedm/zed_node/odom",Odometry, self.orientation_zed_callback)
        #rospy.Subscriber("/gazebo/model_states",ModelStates, self.orientation_callback)
        pub = rospy.Publisher('/Face_recognition/local_position', Pose, queue_size=10)
        landing_pub = rospy.Publisher('/Face_recognition/landing', String, queue_size=10)

        self.takeoff = ''
        self.takeoff_flag = False
        self.landing_flag = False
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
                # rospy.loginfo("Count equal to: %f" % self.count)
                rospy.loginfo("Takingoff...")
                self.pose.position.x = 0.0
                self.pose.position.y = 0.0
                self.pose.position.z = 0.70
                self.pose.orientation.x = 0  # -0.018059104681
                self.pose.orientation.y = 0  # 0.734654724598
                self.pose.orientation.z = 0  # 0.00352329877205
                self.pose.orientation.w = 1  # 0.678191721439
                pub.publish(self.pose)
                rospy.loginfo("altitude equal to: %f" % self.pose_pos_z)
                # self.zVal += 0.01

                if self.pose_pos_z >= 0.68:
                    rospy.loginfo("altitude GAZEBO to: %f" % self.pose_pos_z)
                    self.takeoff_flag = False
                    self.landing_flag = True
                    sleep(15)
                
            
            if self.takeoff_flag==False and self.landing_flag==True:
                
                rospy.loginfo("Landing...")
                self.pose.position.x = 0.0
                self.pose.position.y = -0.0
                self.pose.position.z = 0.30
                self.pose.orientation.x = 0  # -0.018059104681
                self.pose.orientation.y = 0  # 0.734654724598
                self.pose.orientation.z = 0  # 0.00352329877205
                self.pose.orientation.w = 1  # 0.678191721439
                #print(self.zVal)
                pub.publish(self.pose)
                sleep(.3)
                rospy.loginfo("altitude equal to: %f" % self.pose.position.z)
                self.pose.position.x = 0.0
                self.pose.position.y = -0.0
                self.pose.position.z = 0.12
                self.pose.orientation.x = 0  # -0.018059104681
                self.pose.orientation.y = 0  # 0.734654724598
                self.pose.orientation.z = 0  # 0.00352329877205
                self.pose.orientation.w = 1  # 0.678191721439
                pub.publish(self.pose)

                if self.pose_pos_z <= 0.18:
                    rospy.loginfo("altitude GAZEBO to: %f" % self.pose_pos_z)
                    self.takeoff_flag = False
                    self.landing_flag = False
                    msg = "Landing"
                    landing_pub.publish(msg)
                    break

            rate.sleep()


if __name__ == "__main__":
    rospy.init_node('takingoff_node', anonymous=True)
    rospy.loginfo("Takeoff_landing node ready")
    try:
        drone_class = drone_1()
    except rospy.ROSInterruptException:
        raise





"""
if self.takeoff_flag==False and self.landing_flag==True:
                sleep(15)
                rospy.loginfo("Landing...")
                self.pose.position.x = 0.0
                self.pose.position.y = -0.0
                self.pose.position.z = 0.40
                self.pose.orientation.x = 0  # -0.018059104681
                self.pose.orientation.y = 0  # 0.734654724598
                self.pose.orientation.z = 0  # 0.00352329877205
                self.pose.orientation.w = 1  # 0.678191721439
                sleep(.3)
                pub.publish(self.pose)
                sleep(4)
                self.pose.position.x = 0.0
                self.pose.position.y = -0.0
                self.pose.position.z = 0.20
                self.pose.orientation.x = 0  # -0.018059104681
                self.pose.orientation.y = 0  # 0.734654724598
                self.pose.orientation.z = 0  # 0.00352329877205
                self.pose.orientation.w = 1  # 0.678191721439
                sleep(.3)
                pub.publish(self.pose)
                sleep(2)
                self.pose.position.x = 0.0
                self.pose.position.y = -0.0
                self.pose.position.z = 0.15
                self.pose.orientation.x = 0  # -0.018059104681
                self.pose.orientation.y = 0  # 0.734654724598
                self.pose.orientation.z = 0  # 0.00352329877205
                self.pose.orientation.w = 1  # 0.678191721439
                sleep(.3)
                pub.publish(self.pose)
                sleep(3)
                msg = "Landing"
                landing_pub.publish(msg)
                sleep(2)
                break
"""
