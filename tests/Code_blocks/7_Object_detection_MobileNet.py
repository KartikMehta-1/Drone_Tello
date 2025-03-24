import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np
from djitellopy import tello
from time import sleep
import time
import sys
sys.path.append('D:\Kartik\learning\AI_Projects\Object_Detection_Models\Pretrained_Models')
from MobileNet_Module import MobileNetSSD

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

model_path = 'D:\Kartik\learning\AI_Projects\Object_Detection_Models\Pretrained_Models\MobileNetSSD_Caffe\mobilenet_iter_73000.caffemodel'
prototxt_path = 'D:\Kartik\learning\AI_Projects\Object_Detection_Models\Pretrained_Models\MobileNetSSD_Caffe\deploy.prototxt'

model = MobileNetSSD(prototxt_path,model_path, confidence_threshold=0.8)

while True:
    img = me.get_frame_read().frame #get frame from drone

    # Perform object detection
    detections = model.detect_objects(img)

    #Draw bbox of detected objects
    frame_with_detections = model.draw_detections(img, detections)
    if detections: 
        # Update the set of detected objects
        detected_objects.add(detections[0][0])

    # Display the output frame
    cv2.imshow("Object Detection", frame_with_detections)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# Print the final set of detected objects
print("Detected objects:", detected_objects)