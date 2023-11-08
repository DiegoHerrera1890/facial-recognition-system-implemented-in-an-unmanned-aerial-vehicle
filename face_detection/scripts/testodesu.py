import cv2
import numpy as np
import csv

'''
Faces = np.array([[234,162,78,78],[261,356,77,77]])
print("Array faces is:\n", Faces)
print('Faces type: ', type(Faces))
print('Faces shape: ', Faces.shape)
print('Lenght face: ', len(Faces))
print("Array faces is:\n", Faces[0])

for x, y, w, h in Faces:
	Area = []
	A = h*w
	print("Area: ", A)


with open('/media/jetson/B2E4-69EA2/Face_recognition_data_analysis/min_distance_data.csv', 'w', newline='') as f:
	write_csv = csv.writer(f)
	#write_csv.writerow(['col1', 'col2', 'col3'])
	for a in range(20):
		write_csv.writerow(['col1', 'col2', 'col3'])
'''

def noidea(boxes_ids,olap,X1,Y1,X2,Y2,font,img):
	if len(boxes_ids) >1 and len(olap)==len(boxes_ids):
		print("len olap", len(olap))
		print("if bucle", olap[len(olap)-1])
		id_New=olap[len(olap)-1]
		print("coordinates of Id 2: ", X1,Y1,X2,Y2)
		cv2.putText(img, str(id_New), (X1,Y1 - 15), font, 1, (255, 0, 0), 1)
		cv2.rectangle(img, (X1, Y1), (X2, Y2), (255, 0, 0), 2)

	elif len(boxes_ids) ==1:
		id_New=olap[len(olap)-1]
		print("coordinates of Id 2: ", X1,Y1,X2,Y2)
		cv2.putText(img, str(id_New), (X1,Y1 - 15), font, 1, (255, 0, 0), 1)
		cv2.rectangle(img, (X1, Y1), (X2, Y2), (255, 0, 0), 2)


