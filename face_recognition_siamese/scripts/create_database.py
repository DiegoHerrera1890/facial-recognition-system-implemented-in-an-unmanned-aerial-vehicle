import numpy as np
import cv2
import os  # Add this import
import csv
import sys
from time import sleep


face_cascade = cv2.CascadeClassifier('/home/xonapa/drone_ws/src/face_detection/scripts/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(-1)

save_path = '/media/xonapa/3A3A-CE50/faces'  # Set the path where you want to save the face images
if not os.path.exists(save_path):
    os.makedirs(save_path)

count = 0  # Counter for file names

while True:
    ret, img = cap.read()
    height, width, channels = img.shape
    faces = face_cascade.detectMultiScale(img, 1.3, 9)

    if len(faces) > 0:
        font = cv2.FONT_HERSHEY_DUPLEX
        for (x, y, w, h) in faces:
            c1 = x + (w // 2)
            c2 = y + (h // 2)
            A = h * w
            roi = img[y-2:y + h+2, x-2:x + w+2]
            roi = cv2.resize(roi, (224, 224))
            
            # Save the ROI as a JPG file
            filename = os.path.join(save_path, f"diego_{count:1d}.jpg")
            cv2.imwrite(filename, roi)
            count += 1
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    else:
        print("No faces detected")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', img)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

