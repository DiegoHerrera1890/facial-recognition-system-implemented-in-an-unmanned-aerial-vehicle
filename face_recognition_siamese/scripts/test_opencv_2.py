#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from std_msgs.msg import String
import sys
#from face_rec.msg import coordinates
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

BASE_DIR = "/home/jetson/catkin_ws/src/face_recognition_keras/scripts"

'''
#import matplotlib.pyplot as plt
# %matplotlib inline  # if you are running this code in Jupyter notebook
#fig = plt.figure(figsize=(10, 7))
#rows = 2
#columns = 2
# reads image 'opencv-logo.png' as grayscale

img = cv2.imread("/home/jetson/catkin_ws/src/face_recognition_keras/scripts/images/biden.jpg", 0)
#print('Shape: ', img.shape)
#img1 = img[...,::-1] # img[...,::-1]   
#print('Shape: ', img1.shape)
cv2.imshow("Original", img)
#cv2.imshow("Modified", img1)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''



'''
import rospy
import sys
import numpy as np
import cv2 as cv
cap = cv.VideoCapture(-1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()


'''


bridge = CvBridge()
def cam_test():                         
  face_cascade = cv2.CascadeClassifier(BASE_DIR + '/haarcascade_frontalface_default.xml')
  cap = cv2.VideoCapture(-1)
  rospy.init_node('talker', anonymous=True)
  pub = rospy.Publisher('image_data', String, queue_size=10)
  rate = rospy.Rate(10)
  while True:
      ret, img = cap.read()
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.3, 5)
      cv2.circle(img,(320,240),15,(0,0,255),2)
      for (x,y,w,h) in faces:
          cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
          c1 =x+(w/2)
          c2 =y+(h/2)
          print("C1: ", c1)
          print("INt C1: ", int(c1))
          print("C2: ", c2)
          cv2.circle(img,(int(c1),int(c2)),15,(0,255,0),2)
          roi_gray = gray[y:y+h, x:x+w]
          roi_color = img[y:y+h, x:x+w]
      bridge = CvBridge()
      msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
      msg.header.stamp = rospy.Time.now()
      msg.header.frame_id = "opencv_rviz_frame"
      pub.publish(msg)
      rate.sleep()      
      '''
      cv2.imshow('img',img)
      k = cv2.waitKey(30) & 0xff
      if k == 27:
          break
      
  cap.release()
  cv2.destroyAllWindows()
      '''

if __name__ == '__main__':
       try:
           cam_test()
       except rospy.ROSInterruptException:
           pass
  










































