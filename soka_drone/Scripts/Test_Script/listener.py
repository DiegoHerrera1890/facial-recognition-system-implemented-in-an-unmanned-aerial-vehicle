#!/usr/bin/env python3
'''
Streaming T265 fisheye camera and recording video
December 2, 2021, 16:59
@DiegoHerrera
'''

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class t265_fisheye():
    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg)
        frame = np.array(frame, dtype=np.uint8)
        self.video_out.write(frame)
        cv2.imshow('Fisheye_1', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("Streaming finished")

    def image_callback_2(self, msg_2):
        frame_2 = self.bridge.imgmsg_to_cv2(msg_2)
        #frame = np.array(frame, dtype=np.uint8)
        cv2.imshow('fisheye2', frame_2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown("Streaming finished")

    def cleanup(self):
        print("Shutting down T265 streaming.")
        cv2.destroyAllWindows()   

    def __init__(self):
        rospy.on_shutdown(self.cleanup)
        self.bridge = CvBridge()
        self.frame_width = 480
        self.frame_height = 320
        self.fourcc = cv2.VideoWriter_fourcc('I','4','2','0')
        self.video_out = cv2.VideoWriter('Fisheye_1.avi', self.fourcc, 10, (self.frame_width,self.frame_height))
        self.image_sub = rospy.Subscriber("/camera/fisheye1/image_raw", Image, self.image_callback)
        self.image_sub = rospy.Subscriber("/camera/fisheye2/image_raw", Image, self.image_callback_2)
        
        rospy.loginfo("Waiting for T265 topic...")


if __name__ == '__main__':
    try:
        rospy.init_node('T265_Fisheye', anonymous=True)
        t265_fisheye()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.DestroyAllWindows()
        pass