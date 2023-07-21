while not rospy.is_shutdown():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceRects = face_cascade.detectMultiScale(gray, 1.3, 5)
    detections = []
    detections2 = ()
    if len(faceRects) > 0:
        rospy.loginfo("Face found...")
        for (x, y, w, h) in faceRects:
            c1 =x+(w//2) # Center of BBox X
            c2 =y+(h//2) # Center of BBox Y
            A = h*w  # Area of Bounding box
            #area = h*w  # Area of Bounding box
            #face_list_area.append(area)
            coordinates = Point(x=c1, y=A, z=c2)
            roi = frame[y:y + h, x:x + w]
            roi = cv2.resize(roi, (96, 96)) 
            #roi2 = cv2.imread(path)
            min_dist, identity = who_is_it(roi, database, FRmodel)
            average_value.img_writer(img_count, roi)
            average_value.img_value_writer(img_count,min_dist, identity)
            avg_val, flag,total_avg, identityy = average_value.avg_dist(j, min_dist, total_sum, identity, avg_val,flag=False)
            total_sum = total_avg
            avg_val = round(avg_val,2)

            img_count += 1
            j+=1
            if j== 5:
                j=0
                total_sum=0        

            if flag==True and avg_val > 0.53:
                #print("average value is: ", avg_val)
                name = 'Unknown'
                id_unknown = id_N
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (x,y - 15), font, 1, (255, 0, 0), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            elif flag==True and avg_val <= 0.53:
                name = identityy
                print("This person seems to be: ",name)
                id_known = id_N
                #print("average value is: ", avg_val)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (x,y - 15), font, 1, (255, 0, 0), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                 