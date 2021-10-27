#!/usr/bin/env python
'''
This code is to control the position of the drone in the Z axis
Author: @Diego Herrera
email: alberto18_90@outlook.com
'''
import cv2
import rospy
import mavros
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
#from gazebo_msgs.msg import ModelStates
from tf.transformations import quaternion_from_euler
from time import sleep
#global flag
#flag = False
# Callback method for state subscriber
current_state = State()  # Reading the current state from mavros msgs

mavros.set_namespace()

class drone_control:
    def __init__(self):
        self.pose = PoseStamped()
        self.local_position_publisher = rospy.Publisher(mavros.get_topic('setpoint_position', 'local'), PoseStamped,
                                           queue_size=10)  #
        self.sub = rospy.Subscriber("/Face_recognition/local_position", Pose, coordinates_orientation_2)
        self.state_subscriber = rospy.Subscriber(mavros.get_topic('state'), State, self.state_callback)
        self.arming_client = rospy.ServiceProxy(mavros.get_topic('cmd', 'arming'), CommandBool)
        self.takeoff_client = rospy.ServiceProxy(mavros.get_topic('cmd', 'takeoff'), CommandTOL)
        #landing_client = rospy.ServiceProxy(mavros.get_topic('cmd', 'land'), CommandTOL)
        self.set_mode_client = rospy.ServiceProxy(mavros.get_topic('set_mode'), SetMode)
        


    def state_callback(self, state):
    self.current_state = state


    def coordinates_orientation_2(self, data_2):
        # print('data of Face Found: ', data_2)
        self.pose.header.stamp = rospy.Time.now()
        self.pose.pose.position.x = data_2.position.x
        self.pose.pose.position.y = data_2.position.y
        self.pose.pose.position.z = data_2.position.z
        # quat = quaternion_from_euler(rVal2, pVal2, yawVal2) # + pi_2)
        self.pose.pose.orientation.x = data_2.orientation.x  # quat[0]
        self.pose.pose.orientation.y = data_2.orientation.y  # quat[1]
        self.pose.pose.orientation.z = data_2.orientation.z  # quat[2]
        self.pose.pose.orientation.w = data_2.orientation.w  # quat[3]
        # pose.header.stamp = rospy.Time.now()
        # print("Pose: ", pose.pose.position)
        # local_position_publisher.publish(pose)

    def land_callback(self, msg):
        #global flag
        flag = True


def position_control():
    print("Position control function")
    # rospy.init_node('Offboard_node', anonymous=True)
    prev_state = self.current_state
    rate = rospy.Rate(20.0)

    # Sending a few points before start
    for i in range(100):
        local_position_publisher.publish(pose)
        rate.sleep()

    # We need to wait for FCU connection

    while not current_state.connected:
        rate.sleep()

    self.last_request = rospy.get_rostime()

    while not rospy.is_shutdown():
        now = rospy.get_rostime()
        # qxVal, yVal, zVal, rVal, pVal, yawVal = 0 , 0, 1.6, 0, 0, 0
        if current_state.mode != "OFFBOARD" and (now - last_request > rospy.Duration(5.)):
            set_mode_client(base_mode=0, custom_mode="OFFBOARD")
            last_request = now

        else:
            if not current_state.armed and (now - last_request > rospy.Duration(5.)):
                arming_client(True)
                last_request = now

        if current_state.armed:
            print("Drone ready to fly")
        if prev_state.armed != current_state.armed:
            rospy.loginfo("Vehicle armed: %r" % current_state.armed)
        if prev_state.mode != current_state.mode:
            rospy.loginfo("Current mode: %s" % current_state.mode)

        prev_state = current_state
        
        
        # rospy.Subscriber("/Face_recognition/drone_search", Pose, coordinates_orientation)
        pose.header.stamp = rospy.Time.now()
        local_position_publisher.publish(pose)
        rospy.Subscriber("/Face_recognition/landing", String, land_callback)
        if flag:
            print("hello landing")
            set_mode_client(base_mode=0, custom_mode="AUTO.LAND")
            sleep(8)
            flag = False
            break
        rate.sleep()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    try:
        rospy.init_node('Quadcopter_node', anonymous=True)
        position_control()
    except rospy.ROSInterruptException:
        
        pass
