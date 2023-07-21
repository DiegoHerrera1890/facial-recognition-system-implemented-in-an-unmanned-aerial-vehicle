#!/usr/bin/env python3

'''
Main node for RPi slave
Subscribe to master RPi and publish to Arduino due.
@DiegoHerrera
'''

import rospy
import sys
#import RPi.GPIO as gpio
import gpio
from time import sleep

class data_processing():        

    #def angle_callback(self, msg):
    #    self.yaw_angle = msg.data
    #    return self.yaw_angle
  

    def __init__(self):
        
        #self.sub = rospy.Subscriber("/Face_recognition/face_found", String, self.face_found_callback)
        #pub = rospy.Publisher('/Face_recognition/coordinates', Pose, queue_size=10)
        #self.d = rospy.Duration(0.1)
        while True:
            gpio.set(298,0)
            '''
            sleep(.4)
            gpio.set(480,0)
            gpio.set(388,0)
            gpio.set(298,0)
            sleep(.4)
            gpio.set(480,1)
            gpio.set(388,1)
            gpio.set(298,1)
            sleep(.2)
            gpio.set(388,0)
            sleep(.2)
            gpio.set(388,1)
            sleep(.2)
            gpio.set(298,0)
            sleep(.2)
            gpio.set(298,1)
            sleep(.2)
            gpio.set(480,0)
            sleep(.2)
            gpio.set(480,1)
            sleep(.2)
            gpio.set(480,0)
            gpio.set(388,0)
            gpio.set(298,0)
            sleep(.2)
            gpio.set(480,1)
            gpio.set(388,1)
            gpio.set(298,1)
            sleep(.4)
            gpio.set(480,0)
            gpio.set(388,0)
            gpio.set(298,0)
            sleep(.4)
            gpio.set(480,1)
            gpio.set(388,1)
            gpio.set(298,1)
            sleep(.4)
            gpio.set(480,0)
            gpio.set(388,0)
            gpio.set(298,0)
            sleep(.4)
            gpio.set(480,1)
            gpio.set(388,1)
            gpio.set(298,1)
            '''
            #rospy.sleep(self.d)
            


if __name__ == "__main__":
    rospy.init_node('Blinking', anonymous=True)
    gpio.setup(388,gpio.OUT)
    gpio.setup(298,gpio.OUT)
    gpio.setup(480,gpio.OUT)
    gpio.set(388,1)
    gpio.set(298,1)
    gpio.set(480,1)
    rospy.loginfo("Blinking node ready")
    rate = rospy.Rate(10)  # 10hz
    blinking_class = data_processing()
    rospy.spin()