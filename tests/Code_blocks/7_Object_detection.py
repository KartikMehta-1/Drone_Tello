import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np
from djitellopy import tello
from time import sleep
import time
import sys

#Initiate Tello
me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()
#me.takeoff()
#me.send_rc_control(0,0,25,0)

#Define image output size
w,h = 300,300

# Initialize an empty set to keep track of detected objects
detected_objects = set()

while True:
    img = me.get_frame_read().frame #get frame from drone
    img = cv2.resize(img,(w,h))

    # Perform object detection
    bbox, label, conf = cv.detect_common_objects(img)

    # Update the set of detected objects
    detected_objects.update(label)

    # Draw bounding boxes around detected objects
    output_frame = draw_bbox(img, bbox, label, conf)

    # Display the output frame
    cv2.imshow("Object Detection", output_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# Print the final set of detected objects
print("Detected objects:", detected_objects)