import cv2
import keyboard
import numpy as np
import sys
import time
import os
from datetime import datetime 
from djitellopy import tello
import threading
from threading import Lock
import json
import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format
    handlers=[
        logging.FileHandler("D:/Kartik/learning/AI_Projects/Drone_Python_Control/drone_program.log"),  # Save logs to a file
        logging.StreamHandler()  # Also output logs to the console
    ]
)
# Load configuration
config_path = "D:/Kartik/learning/AI_Projects/Drone_Python_Control/config.json"
with open(config_path, "r") as config_file:
    config = json.load(config_file)
#Import module paths
handgesture_module_path = config["modules"]["handgesture_module_path"]
pose_estimation_module_path = config["modules"]["pose_estimation_module_path"]
mobilenet_module_path = config["modules"]["mobilenet_module_path"]
#Add paths to sys.path
sys.path.append(handgesture_module_path)
sys.path.append(pose_estimation_module_path)
sys.path.append(mobilenet_module_path)
#Import modules
from Handgesture_module import HandGesture
import PoseEstimationModule as pem
from MobileNet_Module import MobileNetSSD
import speech_recognition as sr
import pyttsx3

#Load the mode of operation - drone / webcamera
mode = config["modes"]["mode"]

#Provide links to required AI models & labels
handGesture_Model_path = config["paths"]["handGesture_Model_path"]
handGesture_Label_path = config["paths"]["handGesture_Label_path"]
obj_det_model_path = config["paths"]["obj_det_model_path"]
obj_det_proto_path = config["paths"]["obj_det_proto_path"]

#Call hyperparameters
hand_gesture_confidence = config["hyperparameters"]["hand_gesture_detection_confidence"]
object_detection_confidence = config["hyperparameters"]["object_detection_confidence"]
pose_detection_confidence = config["hyperparameters"]["pose_min_detection_confidence"]
pose_tracking_confidence = config["hyperparameters"]["pose_min_tracking_confidence"]

#Initialize drone settings
state = config["drone_settings"]["state"]
command = config["drone_settings"]["command"]
direction = config["drone_settings"]["direction"]
image_mode = config["drone_settings"]["image_mode"]
video_mode = config["drone_settings"]["video_mode"]
speedparam = config["drone_settings"]["drone_speed"]
max_altitude = config["drone_settings"]["max_altitude"]
min_battery = config["drone_settings"]["min_battery_threshold"]
vals = config["drone_settings"]["vals"] #lr,fb,ud,yv

#List of keys which are used to command drone
key_CODES = config["key_CODES"] # Shift, Up, Down, Left, Right, 0, 1, 2, 3, 4, 9, 0, q, -, =, p, i, o

# Screen dimensions
w = config["screen_dimensions"]["width"]
h = config["screen_dimensions"]["height"]

#pid controls
shLenRange = config["pid_control_settings"]["shLenRange"]
pid = config["pid_control_settings"]["pid"]
perror = config["pid_control_settings"]["perror"]

#Other variables
pTime = config["fps_control_settings"]["pTime"]
cTime = config["fps_control_settings"]["cTime"]
speak_lock = Lock()  # To ensure thread safety
speak_cancel_flag = False  # Global flag to cancel speech
speak_thread = None
detected_objects = set()

#Load AI models
handgesture_model = HandGesture(model_path=handGesture_Model_path, label_path=handGesture_Label_path, gesture_confidence=hand_gesture_confidence)
objdetection_model = MobileNetSSD(obj_det_proto_path,obj_det_model_path, confidence_threshold=object_detection_confidence)
pose_detector = pem.poseEstimator(min_detection_confidence=pose_detection_confidence, min_tracking_confidence=pose_tracking_confidence)

# Load Text-to-Speech engine
engine = pyttsx3.init()

    if mode == "drone":
        me.streamoff()
        me.end()
    if mode == "webcam":
        webcam.release()
    cv2.destroyAllWindows()
    logging.info("Drone program terminated.")