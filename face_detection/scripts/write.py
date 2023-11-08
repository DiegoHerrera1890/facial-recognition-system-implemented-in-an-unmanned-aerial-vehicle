#!/usr/bin/env python
import cv2

# Open the camera
cap = cv2.VideoCapture(-1) 

# Define the number of photos to capture
num_photos = 10
photo_count = 0

# Loop to capture photos
while photo_count < num_photos:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    if ret:
        # Display the frame
        cv2.imshow('Camera', frame)
        
        # Wait for 'Space' key press to capture the photo
        if cv2.waitKey(1) == ord(' '):
            # Save the photo
            photo_filename = 'photo_{}.jpg'.format(photo_count)
            cv2.imwrite(photo_filename, frame)
            print('Photo {} captured!'.format(photo_count + 1))
            photo_count += 1
    
    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

