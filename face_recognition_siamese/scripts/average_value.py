#!/usr/bin/env python3
import rospy
import cv2
import csv
from geometry_msgs.msg import Point
from std_msgs.msg import String

pub_Searching = rospy.Publisher('/Face_recognition/Searching', String, queue_size=10)
pub_face_coordinates = rospy.Publisher('/Face_recognition/face_coordinates', Point, queue_size=10)
pub_face_matching = rospy.Publisher('/Face_recognition/face_matching', String, queue_size=10)
# pub_no_match = rospy.Publisher("/Face_recognition/face_notmatch", String, queue_size=10)



def write_image(count, roi):
    if count < 200:
        print("done")
		

def write_image_value(data, min_dist, identity):
    if data < 100:
        with open('/media/xonapa/3A3A-CE50/Face_recognition_data_analysis/1_stage_avg_dist_data.csv', 'a', newline='') as f:
            write_csv = csv.writer(f)
            write_csv.writerow([min_dist, identity])
        print("done")
        

def calculate_average_distance(msg, value, total, identity, avg_val, flag):
    if msg < 2:
        numbers = value
        total += numbers
        # print("total: ", total)
        if msg == 1:
            avg_val = total / 2
            # print("average value is: ", avg_val)
            flag = True
            '''
            with open('/media/xonapa/3A3A-CE50/Face_recognition_data_analysis/2_stage_avg_dist_data', 'a', newline='') as f:
                write_csv = csv.writer(f)
                write_csv.writerow([avg_val, identity])
            '''            
            msg = 0
    return avg_val, flag, total, identity



def new_algorithm(coordinates, X1, Y1, X2, Y2, flag, avg_val, id_N, A_dict, B_dict, identityy, name, frame, id_known,
                  id_unknown, flag_nf, flag_ff):
    rospy.loginfo("Average is: %f", avg_val)
    coord_center = Point()
    coord_center.x = coordinates.x
    CC1 = coord_center.x
    if flag == True and avg_val > 0.59:
        id_unknown = id_N
        rospy.loginfo("Average more than: %f", 0.59)
        for value_1 in list(A_dict.values()):
            if value_1 == id_unknown and not flag_nf:
                flag_ff = True
                flag_nf = False
                name = identityy
                A_dict[name] = id_known
                color_rgb = (255, 0, 0)
                font = cv2.FONT_HERSHEY_DUPLEX
                
                with open('/media/xonapa/3A3A-CE50/Face_recognition_data_analysis/2_known_stage_dist_data.csv', 'a', newline='') as f:
                    write_csv = csv.writer(f)
                    write_csv.writerow([avg_val, identityy])
                
                rospy.loginfo("Person ID: %s", name)
                face_message = "face_match"
                rospy.loginfo("face_match")
                pub_face_matching.publish(face_message)
                pub_face_coordinates.publish(coordinates)
            elif  not flag_ff:
                name = 'Unknown'
                B_dict.clear()  # clean B dictionary
                B_dict[name] = id_unknown  # unknown
                flag_ff = False
                flag_nf = True
                color_rgb = (0, 0, 255)
                font = cv2.FONT_HERSHEY_DUPLEX
                
                with open('/media/xonapa/3A3A-CE50/Face_recognition_data_analysis/2_unknown_stage_dist_data.csv', 'a', newline='') as f:
                    write_csv = csv.writer(f)
                    write_csv.writerow([avg_val, identityy])
                
                rospy.loginfo("Person ID: %s", name)
                face_message = "Unknown"
                search_msg = "Searching"
                pub_face_matching.publish(face_message)
                pub_Searching.publish(search_msg)

            rospy.loginfo("Flag FF: %s", flag_ff) 
            rospy.loginfo("Flag NF: %s", flag_nf) 
            cv2.putText(frame, str(CC1), (X1, Y1 - 45), font, 1, color_rgb, 1)
            cv2.putText(frame, name, (X1, Y1 - 15), font, 1, color_rgb, 1)
            cv2.rectangle(frame, (X1, Y1), (X2, Y2), color_rgb, 2)

    elif flag == True and avg_val <= 0.59:
        #rospy.loginfo("Person ID: %s", name)
        rospy.loginfo("Average less than: %f", 0.59)
        id_known = id_N
        for value_2 in list(B_dict.values()):
            if value_2 == id_known and not flag_ff:
                name = 'Unknown'
                rospy.loginfo("Person ID: %s", name)
                flag_ff = False
                flag_nf = True
                B_dict.clear()  # clean B dictionary
                B_dict[name] = id_unknown  # unknown
                color_rgb = (0, 0, 255)
                font = cv2.FONT_HERSHEY_DUPLEX
                
                with open('/media/xonapa/3A3A-CE50/Face_recognition_data_analysis/2_unknown_stage_dist_data.csv', 'a', newline='') as f:
                    write_csv = csv.writer(f)
                    write_csv.writerow([avg_val, identityy])
                
                face_message = "Unknown"
                search_msg = "Searching"
                pub_face_matching.publish(face_message)
                pub_Searching.publish(search_msg)
            elif  not flag_nf:
                A_dict.clear()  # clean A dictionary
                flag_ff = True
                flag_nf = False
                name = identityy
                A_dict[name] = id_known  # unknown
                color_rgb = (255, 0, 0)
                font = cv2.FONT_HERSHEY_DUPLEX
                
                with open('/media/xonapa/3A3A-CE50/Face_recognition_data_analysis/2_known_stage_dist_data.csv', 'a', newline='') as f:
                    write_csv = csv.writer(f)
                    write_csv.writerow([avg_val, identityy])
                
                face_message = "face_match"
                rospy.loginfo("Person ID: %s", name)
                rospy.loginfo("face_match")
                pub_face_matching.publish(face_message)
                pub_face_coordinates.publish(coordinates)
            cv2.putText(frame, name, (X1, Y1 - 15), font, 1, color_rgb, 1)
            cv2.rectangle(frame, (X1, Y1), (X2, Y2), color_rgb, 2)