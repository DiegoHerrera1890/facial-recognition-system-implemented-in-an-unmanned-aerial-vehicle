#!/usr/bin/env python
"""
Created on Sat Jan 30 13:23:20 2021
Publisher and Subscriber using MavRos
This node is for sending data to Mavros bidirectional to control
the drone movement
@author: Diego Herrera
"""
import rospy
import tf
from geometry_msgs.msg import Pose, PoseStamped, Twist, Quaternion
from mavros_msgs.msg import OverrideRCIn
from mavros_msgs.msg import RCIn
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool
from mavros_msgs.srv import SetMode
from mavros_msgs.srv import CommandTOL
from mavros_msgs.srv import CommandHome

pi_2 = 3.141592654 / 2.0


class mavros_control():
    def __init__(self):
        rospy.init_node("Offboard_Node")
        # Subscribe from mavros to get the pose stamp and rc data.
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.pose_callback)
        rospy.Subscriber("/mavros/rc/in", RCIn, self.rc_callback)
        rospy.Subscriber("/mavros/state", State, self.state_callback)

        self.position_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.vel_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=10)
        self.rc_override_pub = rospy.Publisher("/mavros/rc/override", OverrideRCIn, queue_size=10)
        # Flight modes for PX4
        # STABILIZE
        # OFFBOARD
        # AUTO.LAND
        self.set_home_position = rospy.ServiceProxy("/mavros/cmd/set_home", CommandHome)
        self.set_mode_service = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        self.arming_service = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.takingoff_service = rospy.ServiceProxy("/mavros/cmd/takeoff", CommandTOL)

        self.rc = RCIn()
        self.pose = Pose()
        self.timestamp = rospy.Time()
        self.state = State()

    def pose_callback(self, data):
        """
        Manipulate the local position information
        """
        self.timestamp = data.header.stamp
        self.pose = data.pose

    def rc_callback(self, data):
        """
        Tracking the current values of the manual RC
        """
        self.rc = data

    def state_callback(self, data):
        """
        Read the current state of mavros
        """
        self.state = data

    def arm(self):
        """
        return: vehicle is armed and ready to fly
        """
        return self.arming_service(True)

    def disarm(self):
        """
        return: vehicle is disarmed
        """
        return self.arming_service(False)

    def takeoff(self, height=1.5):
        """
        :param height: Desire altitude to achieve
        :return current_mode: the current mode of the quadrotor
        """
        current_mode = self.set_mode_service(base_mode=0, custom_mode="OFFBOARD")
        rospy.loginfo("Current mode: %s" % self.state.mode)
        self.arm()

        # Takingoff the drone
        self.takingoff_service(altitude=height)

        return current_mode

    def land(self):
        resp = self.set_mode_service(base_mode=0, custom_mode="AUTO.LAND")
        rospy.sleep(5)
        self.disarm()

    def goto_axes_angles(self, x, y, z, ro, pi, ya):
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z

        quat = tf.transformations.quaternion_from_euler(ro, pi, ya + pi_2)
        print("Quaternion", quat)
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        self.goto(pose)

    def goto(self, pose):
        """
        :param pose: Set the target as the next setpoint by sending Local_Position_NED
        """
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.timestamp
        pose_stamped.pose = pose

        self.position_pub.publish(pose_stamped)


def position_control():
    """
    Main function for position control
    """
    control = mavros_control()
    rospy.sleep(2)
    print("arming and taking off")
    control.takeoff()
    rospy.sleep(30)
    control.goto_axes_angles(0, 0, 2.0, 0, 0, 0)
    rospy.sleep(3)

    print("Waypoint 1 (X = 0, Y = 0.4, Z = 1.2)")
    control.goto_axes_angles(0.0, 0.0, 2.5, 0, 0, -1 * pi_2)
    rospy.sleep(2)
    control.goto_axes_angles(0.0, 0.4, 2.0, 0, 0, -1 * pi_2)
    rospy.sleep(5)
    control.goto_axes_angles(0.0, 0.0, 1.5, 0, 0, 0)
    rospy.sleep(4)

    print("Landing")
    control.land()



if __name__ == '__main__':
    try:
        position_control()
    except rospy.ROSInterruptException:
        pass
