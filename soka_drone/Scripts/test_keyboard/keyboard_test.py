#!/usr/bin/env python
#import getch
import rospy
from std_msgs.msg import String  # String message
from std_msgs.msg import Int8
import tty
import sys
import termios



def keys():
    count_X = 0
    count_Y = 0
    count_Z = 0
    orig_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin)
    x = 0
    pub = rospy.Publisher('key', Int8, queue_size=10)  # "key" is the publisher name
    rospy.init_node('keypress', anonymous=True)
    rospy.loginfo("Node ready")
    rate = rospy.Rate(10)  # try removing this line ans see what happens
    while not rospy.is_shutdown():
        #k = cv2.waitKey(0)#ord(getch.getch())  # this is used to convert the keypress event in the keyboard or joypad , joystick to a
        # ord value 
        k=sys.stdin.read(1)[0]
        if k == 'q':
            print(k)
            break
        elif k == 'w':  # Special keys (arrows, f keys, ins, del, etc.)
            rospy.loginfo(str(k))  # to print on  terminal
            count_X += 0.1
            print(count_X)
            pub.publish(k)  # to publish
        elif k == 's':  # Down arrow
            rospy.loginfo(str(k))  # to print on  terminal
            count_X -= 0.1
            print(count_X)
            pub.publish(k)  # to publish
        elif k == 'd':  # Up arrow
            rospy.loginfo(str(k))  # to print on  terminal
            pub.publish(k)  # to publish
        elif k == 'a':  # Up arrow
            rospy.loginfo(str(k))  # to print on  terminal
            pub.publish(k)  # to publish

    termios.tcgetattr(sys.stdin, termios.TCSADRAIN, orig_settings)   
    rate.sleep()


# s=115,e=101,g=103,b=98

if __name__ == '__main__':
    try:
        keys()
    except rospy.ROSInterruptException:
        pass
