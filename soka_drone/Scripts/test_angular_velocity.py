#!/usr/bin/env python

'''
import rospy
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseStamped
import mavros
import numpy as np
from tf.transformations import quaternion_from_euler
from time import sleep

mavros.set_namespace()
pub = rospy.Publisher(mavros.get_topic('setpoint_attitude', 'attitude'), PoseStamped, queue_size=10)
#pub2 = rospy.Publisher(mavros.get_topic('local_position', 'velocity'), Twist, queue_size=10)   
pub2 = rospy.Publisher(mavros.get_topic('setpoint_attitude', 'cmd_vel'), TwistStamped, queue_size=10)

#setpoint_attitude/cmd_vel (geometry_msgs/TwistStamped)
#setpoint_attitude/attitude (geometry_msgs/PoseStamped)
pose = PoseStamped()
velocity = TwistStamped()

def move():
    # Starts a new node
    
    two_pi = 2*3.14159265359
    

    #Receiveing the user's input
    print("Let's move your robot")
    #speed = input("Input your speed:")
    #isForward = input("Foward?: ")#True or False
    yawVal = 0
    #Checking if the movement is forward or backwards
    while True: #if data.data =="Searching"and self.yawVal<two_pi:
        pose.pose.position.x = 8
        pose.pose.position.y = 0
        pose.pose.position.z = 4
        print(pose)
        pub.publish(pose)
        velocity.twist.linear.x = 1.0
        velocity.twist.linear.y = 0.0
        velocity.twist.linear.z = 0.0
        velocity.twist.angular.x = 0.0
        velocity.twist.angular.y = 0.0
        velocity.twist.angular.z = 0.4
        print(velocity)
        pub2.publish(velocity)
        sleep(1)
            
    #else:
    #    self.yawVal = 0 
    

if __name__ == '__main__':
    try:
        rospy.init_node('robot_cleaner', anonymous=True)
        #Testing our function
        move()
    except rospy.ROSInterruptException: pass
    '''

    #!/usr/bin/env python
'''
This code is to control the position of the drone in the Z axis
Author: @Diego Herrera
email: alberto18_90@outlook.com
'''
import cv2
import rospy
import mavros
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler
from time import sleep

# Callback method for state subscriber
current_state = State()  # Reading the current state from mavros msgs


def state_callback(state):
    global current_state
    current_state = state


mavros.set_namespace()
#setpoint_velocity/cmd_vel_unstamped (geometry_msgs/Twist)
local_velocity_publisher = rospy.Publisher(mavros.get_topic('setpoint_velocity', 'cmd_vel_unstamped'), Twist, queue_size=10)
local_position_publisher = rospy.Publisher(mavros.get_topic('setpoint_position', 'local'), PoseStamped, queue_size=10)  #
state_subscriber = rospy.Subscriber(mavros.get_topic('state'), State, state_callback)

arming_client = rospy.ServiceProxy(mavros.get_topic('cmd', 'arming'), CommandBool)
landing_client = rospy.ServiceProxy(mavros.get_topic('cmd', 'land'), CommandTOL)
set_mode_client = rospy.ServiceProxy(mavros.get_topic('set_mode'), SetMode)

pose = PoseStamped()
velocity = Twist()


def position_controller():
    #print('data of Face Found: ', data_2)
    pose.pose.position.x = 0
    pose.pose.position.y = 0
    pose.pose.position.z = 2   
    #quat = quaternion_from_euler(rVal2, pVal2, yawVal2) # + pi_2)
    velocity.linear.x = 0.0
    velocity.linear.y = 0.0
    velocity.linear.z = 0.0
    velocity.angular.x = 0.0
    velocity.angular.y = 0.0
    velocity.angular.z = 0.4
    #pose.header.stamp = rospy.Time.now()
    #print("Pose: ", pose.pose.position)
    #local_position_publisher.publish(pose)
   
  

def position_control():
    print("Position control function")
    #rospy.init_node('Offboard_node', anonymous=True)
    prev_state = current_state
    rate = rospy.Rate(20.0)

    # Sending a few points before start
    for i in range(100):
        local_position_publisher.publish(pose)
        rate.sleep()

    # We need to wait for FCU connection
    while not current_state.connected:
        rate.sleep()

    last_request = rospy.get_rostime()
    
    while not rospy.is_shutdown():
        now = rospy.get_rostime()
        #qxVal, yVal, zVal, rVal, pVal, yawVal = 0 , 0, 1.6, 0, 0, 0
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
        
        position_controller()
        #rospy.Subscriber("/Face_recognition/local_position", Pose, coordinates_orientation_2)
        #rospy.Subscriber("/Face_recognition/drone_search", Pose, coordinates_orientation)
        pose.header.stamp = rospy.Time.now()
        local_position_publisher.publish(pose)
        local_velocity_publisher.publish(velocity)
        #if cv2.waitKey(0) & 0xFF == ord('q'):
        #    landing_client(True) # landing
        #    break
        rate.sleep()


if __name__ == '__main__':
    try:
        rospy.init_node('Quadcopter_node', anonymous=True)
        position_control()
    except rospy.ROSInterruptException:
        pass