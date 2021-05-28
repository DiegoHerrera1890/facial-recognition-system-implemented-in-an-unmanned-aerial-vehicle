import cv2
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 7))
rows = 2
columns = 2
img = cv2.imread('/home/diego/facial-recognition-system-implemented-in-an-unmanned-aerial-vehicle'
                 '/face_recognition_siamese/examples/test_1.jpg')
img2 = cv2.imread('/home/diego/facial-recognition-system-implemented-in-an-unmanned-aerial-vehicle'
                  '/face_recognition_siamese/examples/test_2.jpg')
fig.add_subplot(rows, columns, 1)
# showing image
plt.imshow(img)
plt.axis('off')
plt.title("First")
# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
# showing image
plt.imshow(img2)
plt.axis('off')
plt.title("Second")
plt.show()
