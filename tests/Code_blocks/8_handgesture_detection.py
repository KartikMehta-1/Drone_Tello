import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np
from djitellopy import tello
from time import sleep
import time
import sys
sys.path.append('D:\Kartik\learning\AI_Projects\Object_Detection_Models\OpenCV_Landmark_Based')
from Handgesture_module import HandGesture

#Initiate Tello
me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()
#me.takeoff()
#me.send_rc_control(0,0,25,0)

#Define image output size
w,h = 300,300

model_path = "D:\Kartik\learning\AI_Projects\Object_Detection_Models\OpenCV_Landmark_Based\hand_gesture_model.h5"
label_path = "D:\Kartik\learning\AI_Projects\Object_Detection_Models\OpenCV_Landmark_Based\gesture_classes.npy"

model_trained = HandGesture(model_path=model_path, label_path=label_path, gesture_confidence=0.8)

while True:
    img = me.get_frame_read().frame #get frame from drone
    # Perform object detection
    detections = model_trained.detect_gesture(img)
    # Add labels on camera feed
    if detections:
        print(detections) 
        text = f"Gesture: {detections[0]} ({detections[1]:.2f})"
        cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        speed = 10
        if detections[0] == "ComeNear":
            me.send_rc_control(0,speed,0,0)
            print("Detected move near")
        elif detections[0] == "GoAway":
            me.send_rc_control(0,-speed,0,0)
            print("Detected goaway")
        else:
            me.send_rc_control(0,0,0,0)

    # Display the output frame
    cv2.imshow("Object Detection", img)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()