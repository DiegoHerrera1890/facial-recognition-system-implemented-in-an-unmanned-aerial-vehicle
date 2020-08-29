#!/usr/bin/env python
import numpy as np
import cv2
import rospy
import os
import sys
sys.path.insert(0, '/home/jetson/caffe/python')
import caffe
import matplotlib
matplotlib.use('Agg')
from face_rec.msg import coordinates

def upload_model():
  model = "/home/jetson/catkin_ws/src/face_rec/scripts/vgg_face_deploy.prototxt"
  weights = "/home/jetson/catkin_ws/src/face_rec/scripts/vgg16_iter_30000.caffemodel"
  mean_file = "/home/jetson/catkin_ws/src/face_rec/scripts/train_mean.binaryproto"
  #image_file = "me1.jpg"
  synset_file = "/home/jetson/catkin_ws/src/face_rec/scripts/synset_FR.txt"
  net = caffe.Net(model,weights,caffe.TEST)

  transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

  #converting binaryproto into mean npy
  data=open(mean_file,'rb').read()

  blob=caffe.proto.caffe_pb2.BlobProto()

  blob.ParseFromString(data)
  mean=np.array(caffe.io.blobproto_to_array(blob))[0,:,:,:]

  caffe.set_mode_gpu()

  #image transformation
  transformer.set_mean('data',mean.mean(1).mean(1)) # mean is set
  transformer.set_raw_scale('data', 255) # normalizes the values in the image based on the 0-255 range
  transformer.set_transpose('data', (2,0,1)) # transform an image from (256,256,3) to (3,256,256).
  

#def cam_test():
  # Open stream
  face_cascade = cv2.CascadeClassifier("/home/jetson/catkin_ws/src/face_rec/scripts/haarcascade_frontalface_default.xml")
  stream = cv2.VideoCapture(0)
  pub = rospy.Publisher('chatter', coordinates, queue_size=10)
  rospy.init_node('talker', anonymous=True)
  rate = rospy.Rate(10)
  msg = coordinates()
  

  #loading labels text
  label_mapping = np.loadtxt(synset_file, str, delimiter='\t')

  #semaphore = False
  
  while True:
      ret, frame = stream.read()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.3, 5)
      cv2.circle(frame,(320,240),15,(0,0,255),2)
      preds = []
      for (x,y,w,h) in faces:
          #drawing rectangle
          cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
          #getting detected face
          detected_face = frame[y:y+h, x:x+w]
          print("Detected face shape")
          #print detected_face.shape

    	  #transforming image
          transformed_image = transformer.preprocess('data', detected_face)
	  #print "transformed image shape"
	  #print transformed_image.shape

    	  #model prediction
          net.blobs['data'].reshape(1,*transformed_image.shape)
      	  net.blobs['data'].data[...] = transformed_image
      	  output = net.forward()
      	  pred_probas = output['prob'][0]

          #predicted class
          predicted_class = pred_probas.argmax()
          
          print(' \n ---------------------------')

          #getting name of the user

          user_recognized = label_mapping[predicted_class]
          print("Recognized user: " + user_recognized)
          print '--------------------------- \n'

          cv2.putText(frame,user_recognized,(x+w-120,y), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1)
          c1 =x+(w/2)
          c2 =y+(h/2)
          cv2.circle(frame,(c1,c2),15,(0,255,0),2)
          roi_gray = gray[y:y+h, x:x+w]
          roi_color = frame[y:y+h, x:x+w]
          #print(x)
          msg.X = x+(w/2)
          msg.Y = y+(h/2)


      rospy.loginfo(msg)
      pub.publish(msg)
      rate.sleep() 
      cv2.imshow('frame',frame)
      k = cv2.waitKey(30) & 0xff
      if k == 27:
          break
  stream.release()    
  cv2.destroyAllWindows()

if __name__ == '__main__':
       try:
           upload_model()
       except rospy.ROSInterruptException:
           pass







