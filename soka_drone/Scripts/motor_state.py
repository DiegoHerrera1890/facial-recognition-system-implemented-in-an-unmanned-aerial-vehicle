# ! /usr/bin/env python

import rospy
import mavros
from mavros_msgs.msg import State
from mavros_msgs.msg import ActuatorControl


def actuator_control_callback(msg):
    motor_outputs = msg.controls

    for i, output in enumerate(motor_outputs):
        print("Motor {} output: {}".format(i + 1, output))

def motor_state():
    rospy.Subscriber('/mavros/state', ActuatorControl, actuator_control_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    try:
        rospy.init_node('motor_output_monitor', anonymous=True)
        motor_state()
    except rospy.ROSInterruptException:
        pass