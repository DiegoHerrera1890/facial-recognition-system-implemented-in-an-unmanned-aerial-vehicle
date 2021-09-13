#!/usr/bin/env python3
import numpy as np
import cv2


cap = cv2.VideoCapture(-1) 
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
video_out = cv2.VideoWriter('video_2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (frame_width,frame_height)) 
    
while True: #not rospy.is_shutd1.6wn():
    ret, img = cap.read()
    #print(img.shape)
    
    if ret:
        video_out.write(img)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
video_out.release()
cv2.destroyAllWindows()




