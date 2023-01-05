#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
import csv
from geometry_msgs.msg import Point
from std_msgs.msg import String

numbers = 0
average_face_value = 0

pub_Searching = rospy.Publisher('/Face_recognition/Searching', String, queue_size=10)
pub_face_coordinates = rospy.Publisher('/Face_recognition/face_coordinates', Point, queue_size=10)
pub_face_found = rospy.Publisher('/Face_recognition/face_found', String, queue_size=10)
pub_no_match = rospy.Publisher("/Face_recognition/face_notmatch", String, queue_size=10)

def img_writer(count,roi):
	if count < 200:
		img_name = "/media/xonapa/B2E4-69EA/Face_recognition_data_analysis/images/test_{}.jpg".format(count)
		cv2.imwrite(img_name, roi)

def img_value_writer(data,min_dist, identity):
	if data < 100:
		with open('/media/xonapa/B2E4-69EA/Face_recognition_data_analysis/1_stage_avg_dist_data.csv', 'a', newline='') as f:
			write_csv = csv.writer(f)
			write_csv.writerow([min_dist, identity])
			f.close()

def avg_dist(msg, value, total, identity, avg_val, flag):
	if msg < 2:
		numbers = value
		total += numbers
		#print("total: ", total)
		if msg == 1:
			#print("already 20 values")
			avg_val = total/2
			#print("average value is: ", avg_val)
			flag = True
			with open('/media/xonapa/B2E4-69EA/Face_recognition_data_analysis/2_stage_avg_dist_data', 'a', newline='') as f:
				write_csv = csv.writer(f)
				write_csv.writerow([avg_val, identity])
				f.close()
			msg=0
	return avg_val, flag, total, identity


def new_algorithm(coordinates,X1,Y1,X2,Y2,flag,avg_val,id_N,A_dict,B_dict,identityy,name,frame,id_known,id_unknown):
    
    if flag==True and avg_val > 0.54 :
        id_unknown = id_N
        rospy.loginfo("Person ID: %d", id_unknown)
        rospy.loginfo("Average more than: %d", 0.54)
        for value_1 in list(A_dict.values()):
            if value_1 == id_unknown:
                name = identityy
                A_dict[name] = (id_known)
                color_rgb = (255,0,0)
                font = cv2.FONT_HERSHEY_DUPLEX
                with open('/media/xonapa/B2E4-69EA/Face_recognition_data_analysis/3_kno_stage_dist_data', 'a', newline='') as f:
                    write_csv = csv.writer(f)
                    write_csv.writerow([avg_val, identityy])
                    f.close()
                face_message = "face_found"
                pub_face_found.publish(face_message)
                pub_face_coordinates.publish(coordinates)
            else:
                name = 'Unknown'
                B_dict.clear() # clean B dictionary
                B_dict[name] = (id_unknown) # unknown
                color_rgb = (0,0,255)
                font = cv2.FONT_HERSHEY_DUPLEX
                with open('/media/xonapa/B2E4-69EA/Face_recognition_data_analysis/3_unk_stage_dist_data', 'a', newline='') as f:
                    write_csv = csv.writer(f)
                    write_csv.writerow([avg_val, identityy])
                    f.close()
                face_message = "Unknown"
                search_msg = "Searching"
                pub_face_found.publish(face_message)                
                pub_Searching.publish(search_msg)
            cv2.putText(frame, name, (X1,Y1 - 15), font, 1, color_rgb, 1)
            cv2.rectangle(frame, (X1, Y1), (X2, Y2), color_rgb, 2)

    elif flag==True and avg_val <= 0.54:
        rospy.loginfo("Person ID: %s", name)
        rospy.loginfo("Average less than: %d", 0.54)
        id_known = id_N
        for value_2 in list(B_dict.values()):
            if value_2 == id_known :
                name = 'Unknown'
                B_dict.clear() # clean B dictionary
                B_dict[name] = (id_unknown) # unknown
                color_rgb = (0,0,255)
                font = cv2.FONT_HERSHEY_DUPLEX
                with open('/media/xonapa/B2E4-69EA/Face_recognition_data_analysis/3_unk_stage_dist_data', 'a', newline='') as f:
                    write_csv = csv.writer(f)
                    write_csv.writerow([avg_val, identityy])
                    f.close()
                face_message = "Unknown"
                search_msg = "Searching"
                pub_face_found.publish(face_message)                
                pub_Searching.publish(search_msg)
            else:
                A_dict.clear() # clean A dictionary
                name = identityy
                A_dict[name] = (id_known) # unknown
                color_rgb = (255,0,0)
                font = cv2.FONT_HERSHEY_DUPLEX
                with open('/media/xonapa/B2E4-69EA/Face_recognition_data_analysis/3_kno_stage_dist_data', 'a', newline='') as f:
                    write_csv = csv.writer(f)
                    write_csv.writerow([avg_val, identityy])
                    f.close()
                face_message = "face_found"
                pub_face_found.publish(face_message)
                pub_face_coordinates.publish(coordinates)
            cv2.putText(frame, name, (X1,Y1 - 15), font, 1, color_rgb, 1)
            cv2.rectangle(frame, (X1, Y1), (X2, Y2), color_rgb, 2)