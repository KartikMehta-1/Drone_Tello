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
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format
    handlers=[
        logging.FileHandler("D:/Kartik/Projects/Drone_Tello/data/logs/drone_program.log"),  # Save logs to a file
        logging.StreamHandler()  # Also output logs to the console
    ]
)
# Load configuration
config_path = "D:/Kartik/Projects/Drone_Tello/config/config.json"
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

#Initiate Tello or webcam
if mode == "drone":
    me = tello.Tello()
    try:
        me.connect()
    except Exception as e:
        logging.critical(f"Critical error connecting to drone: {e}")
        exit(1)
    logging.info(f"Drone Battery level: {me.get_battery()}")
    try:
        me.streamon()
    except Exception as e:
        logging.critical(f"Critical error streaming from drone: {e}")
        exit(1)
    me.takeoff()
elif mode == "webcam":
    try:
        webcam = cv2.VideoCapture(0)
    except Exception as e:
        logging.critical(f"Critical error connecting to camera: {e}")
        exit(1)

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
print("Here")

#Load AI models
handgesture_model = HandGesture(model_path=handGesture_Model_path, label_path=handGesture_Label_path, gesture_confidence=hand_gesture_confidence)
objdetection_model = MobileNetSSD(obj_det_proto_path,obj_det_model_path, confidence_threshold=object_detection_confidence)
pose_detector = pem.poseEstimator(min_detection_confidence=pose_detection_confidence, min_tracking_confidence=pose_tracking_confidence)
print("Here")

# Load Text-to-Speech engine
engine = pyttsx3.init()


# Function to process keyboard commands
def process_keyboard_command(pressed_keys, state, command, speedparam, image_mode, video_mode):
    if 11 in pressed_keys:
        state = "idle"
        command ="idle"
        logging.info("State updated to 'idle'")
    elif 2 in pressed_keys:
        state = "det_obj"
        command ="idle"
        logging.info("State updated to 'detect objects'")
        speak("Detecting Objects")
    elif 3 in pressed_keys:
        state = "track_pstr"
        command ="idle"
        logging.info("State updated to 'track_pstr'")
        speak("Tracking pose")
    elif 4 in pressed_keys:
        state = "det_hndgest"
        command ="idle"
        logging.info("State updated to 'det_hndgest'")
        speak("Detecting hand gestures")
    elif 5 in pressed_keys:
        state = "det_face"
        command ="idle"
        logging.info("State updated to 'det_face'")
        speak("Detecting human")
    elif 10 in pressed_keys:
        command ="scan"
    elif 42 in pressed_keys and 75 in pressed_keys:
        command = "move"
    elif 42 in pressed_keys and 77 in pressed_keys:
        command = "move"
    elif 42 in pressed_keys and 72 in pressed_keys:  # Shift + Up
        command = "move"    
    elif 42 in pressed_keys and 80 in pressed_keys:  # Shift + Down
        command = "move"
    elif 72 in pressed_keys:
        command = "move"
    elif 80 in pressed_keys:
        command = "move"
    elif 75 in pressed_keys:
        command = "move"
    elif 77 in pressed_keys:
        command = "move"
    elif 12 in pressed_keys:
        speedparam = speedparam -1
        logging.info(f"Speed updated to {speedparam}")
    elif 13 in pressed_keys:
        speedparam = speedparam +1
        logging.info(f"Speed updated to {speedparam}")
    elif 25 in pressed_keys:
        image_mode = "on"
        logging.info("Taken image")
        speak("Took image")
    elif 23 in pressed_keys:
        video_mode = "create"
        logging.info("Started taking video")
        speak("Started video recording")
    elif 24 in pressed_keys:
        video_mode = "stop"
        logging.info("Finished taking video")
        speak("Stopped video recording")
    return(state, command, speedparam, image_mode, video_mode)

def move(pressed_keys, speed, command, gesture_detected):
    lr,fb,ud,yv = 0,0,0,0
    if (42 in pressed_keys and 75 in pressed_keys) or command == "track" or command == "scan": #shift + left - rotate
        yv = -speed[3]
    if 42 in pressed_keys and 77 in pressed_keys: #shift + right - rotate
        yv = speed[3]
    if (42 in pressed_keys and 72 in pressed_keys) or command == "track" or gesture_detected[0] == "Up":  # Shift + Up
        ud = speed[2]
    if 42 in pressed_keys and 80 in pressed_keys or gesture_detected[0] == "Down":  # Shift + Down
        ud = -speed[2]
    if (42 not in pressed_keys and 72 in pressed_keys) or command == "track" or gesture_detected[0] == "ComeNear": #Forward
        fb = speed[1]
    if (42 not in pressed_keys and 80 in pressed_keys) or gesture_detected[0] == "GoAway": #Behind
        fb = -speed[1]
    if (42 not in pressed_keys and 75 in pressed_keys) or command == "track" or gesture_detected[0] == "Right": #Right for user
        lr = -speed[0]
    if (42 not in pressed_keys and 77 in pressed_keys) or gesture_detected[0] == "Left": #Left for user
        lr = speed[0]
    return [lr,fb,ud,yv]

def handle_command(command, mode, vals, pressed_keys, speed, gesture_detected):
    """Handles the execution of commands."""
    if command == "idle":
        vals = [0, 0, 0, 0]
        if mode == "drone":
            me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    elif command == "move":
        if pressed_keys or gesture_detected[0]:
            vals = move(pressed_keys, speed, command, gesture_detected)
        else:
            command = "idle"
        if mode == "drone":
            me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    elif command == "track":
        vals = move(pressed_keys, speed, command, gesture_detected)
        if mode == "drone":
            me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    elif command == "scan":
        vals = move(pressed_keys, speed, command, gesture_detected)
        if mode == "drone":
            me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    return vals

def handle_state(state, command, img, speed, perror, gesture_detected, detected_objects,pressed_keys, speedparam, image_mode, video_mode):
    """Handles the logic based on the current state."""
    if state == "det_obj":
        detected_objects = detect_obj(img)
    elif state == "track_pstr" and command != "move":
        speed, command, perror = track_shoulder(img, command, speed, perror)
    elif state == "det_hndgest":
        gesture_detected = handgesture_model.detect_gesture(img)
        if gesture_detected:
            command = "move"
        else:
            gesture_detected = ["None", "1"]
            command = "idle"
            if pressed_keys: 
                    state, command, speedparam, image_mode, video_mode = process_keyboard_command(
                        pressed_keys, state, command, speedparam, image_mode, video_mode)
    elif state == "scan":
        command = "scan"
        detected_objects = detect_obj(img)
        if "person" in detected_objects:
            command = "idle"
            speak("Identified a person")
            #delay
            speak("Identifying the person now")
            person = detect_face(img)
            #delay
            speak(f"The person identified is {person}")
            #delay
            speak(f"I will now track {person}")
            #delay
            state = "track_pstr"
    return state, command, False, image_mode, video_mode, gesture_detected

def detect_obj(img):
    detections = objdetection_model.detect_objects(img)
        #Draw bbox of detected objects
    frame_with_detections = objdetection_model.draw_detections(img, detections)
    if detections: 
        # Update the set of detected objects
        if detections[0][0] not in detected_objects:
            print("New object found: ", detections[0][0])
            detected_objects.add(detections[0][0])
            speak(f"I found a {detections[0][0]}")
    return(detected_objects)

def shoulder_center(leftshoulder,rightshoulder):
    x = rightshoulder[1] - leftshoulder[1]
    y = rightshoulder[2] - leftshoulder[2]
    cx = (rightshoulder[1] + leftshoulder[1])/2
    cy = (rightshoulder[2] + leftshoulder[2])/2
    return (cx,cy)

def shoulder_head_dist(leftshoulder,head):
    x = head[1] - leftshoulder[1]
    y = head[2] - leftshoulder[2]
    length = np.sqrt(x*x + y*y)
    return (length)

def track_shoulder(img,command,speed, perror):
    lmList_pose = pose_detector.getPosition(img)
    if len(lmList_pose)!=0:
        if len(lmList_pose[11])!=0 and len(lmList_pose[12])!=0:
            cx, cy = shoulder_center(lmList_pose[11],lmList_pose[12])
            sh_head_dist = shoulder_head_dist(lmList_pose[11],lmList_pose[3])
            if sh_head_dist < shLenRange[0]:
                zerror = int(sh_head_dist - shLenRange[0])
            elif sh_head_dist > shLenRange[1]:
                zerror = int(sh_head_dist - shLenRange[1])
            else: zerror = 0
            error = [0,-zerror, -(cy - h//2)/2,-(cx - w//2)/3]
            for i in range(4):
                if i < len(perror):
                    speed[i] = int(pid[0]*error[i] + pid[1]*(error[i]-perror[i]))
                    speed[i] = int(np.clip(speed[i],-100,100))
            command = "track"
    else: 
        command = "idle"
        error = []
        speed = []
    return(speed, command, error)

def speak(text):
    """Converts text to speech in a parallel thread, replacing the current thread if it's running."""
    global speak_thread, speak_cancel_flag

    def _speak():
        global speak_cancel_flag
        try:
            # Check the cancel flag
            if speak_cancel_flag:
                return  # Skip speaking if the flag is set
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logging.error(f"Error in speak: {e}")
        finally:
            # Ensure the cancel flag is reset
            speak_cancel_flag = False

    with speak_lock:
        # If a thread is already running, set the cancel flag
        if speak_thread and speak_thread.is_alive():
            logging.info("Cancelling previous speech thread.")
            speak_cancel_flag = True

        # Start a new thread for the speak function
        speak_thread = threading.Thread(target=_speak, daemon=True)
        speak_thread.start()

def listen():
    """Captures voice input."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I didn't understand that."

def create_date_folder(base_path="D:/Kartik/Projects/Drone_Tello/data/logs/ride_data"):
    """Create a date-wise folder to store images."""
    # Get the current date
    date_folder = datetime.now().strftime("%Y-%m-%d")

    # Create the base path if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Create the date folder inside the base path
    full_path = os.path.join(base_path, date_folder)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path

def save_frame_to_image(frame, base_path="D:/Kartik/Projects/Drone_Tello/data/logs/ride_data"):
    """Save a given frame to a date-wise folder."""
    # Create a date-wise folder
    folder_path = create_date_folder(base_path)

    # Get the current timestamp for the image filename
    timestamp = datetime.now().strftime("%H-%M-%S")
    file_name = f"image_{timestamp}.jpg"
    file_path = os.path.join(folder_path, file_name)

    # Save the image
    cv2.imwrite(file_path, frame)
    print(f"Image saved at: {file_path}")

def save_video(frame, video_mode, base_path="D:/Kartik/Projects/Drone_Tello/data/logs/ride_data"):
    global out, file_path
    if video_mode == "create":
        # Create a date-wise folder
        folder_path = create_date_folder(base_path)
        # Get the current timestamp for the image filename
        timestamp = datetime.now().strftime("%H-%M-%S")
        file_name = f"image_{timestamp}.avi"
        file_path = os.path.join(folder_path, file_name)
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 20.0
        out = cv2.VideoWriter(file_path, fourcc, fps, (w, h))
        print(f"Recording started. Saving to: {file_path}")
        video_mode = "start"
    if video_mode == "start" and out is not None:
        # Add a red border to the frame
        frame = cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        # Add 'Rec' text to the frame
        cv2.putText(frame, "Rec", (w - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Write the frame to the video file
        out.write(frame)
    if video_mode == "stop" and out is not None:
        # Release the VideoWriter object
        print(f"Video saved at: {file_path}")
        out.release()
        video_mode = "off"
    return video_mode

def handle_media(image_mode, video_mode, img):
    """Handles saving images and videos."""
    if image_mode == "on":
        save_frame_to_image(img)
        image_mode = "off"

    if video_mode in ["create", "start", "stop"]:
        video_mode = save_video(img, video_mode)

    return image_mode, video_mode

def update_and_display_parameters(
    mode, img, state, command, vals, perror, speedparam, gesture_detected, detected_objects, fps, image_mode, video_mode, battery=None
):
    """Updates and displays parameters on the image."""
    content = [f"State: {state}", f"command: {command}", f"rccommand: {vals}", f"fps: {round(fps, 2)}"]
    if mode == "drone":
        if battery and battery < min_battery:
            speak("Low on battery")
        content.extend([
            f"Error: {perror}",
            f"speed value: {speedparam}",
            f"gesture: {gesture_detected}",
            f"objects: {detected_objects}",
            f"image mode: {image_mode}",
            f"video mode: {video_mode}",
            f"battery: {battery}",
        ])
    elif mode == "webcam":
        content.extend([
            f"speed value: {speedparam}",
            f"gesture: {gesture_detected}",
            f"objects: {detected_objects}",
            f"image mode: {image_mode}",
            f"video mode: {video_mode}",
        ])

    line_x, line_y = 10, 40
    line_height = 20
    for text in content:
        cv2.putText(img, text, (line_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        line_y += line_height
    
speak("Hello, lets go on a ride!")

# Main loop
try:
    #Run the while loop for tello for various operations
    while True:
        if mode == "drone":
            img = me.get_frame_read().frame #get frame from drone
            battery = me.get_battery()
        elif mode == "webcam":
            ret, img = webcam.read() 
        h, w, channels = img.shape
        vals = [0,0,0,0]
        line_x, line_y = 10, 40  # Starting point for the text
        line_height = 20  # Line height (spacing between lines)
        if speedparam > 100: speedparam = 100
        elif speedparam < 0: speedparam = 0
        speed = [speedparam,speedparam,speedparam,speedparam]
        gesture_detected = ["None","1"]
        
        #List of keys which are pressed
        pressed_keys = [key for key in key_CODES if keyboard.is_pressed(key)]

        #Determine state and command based on key input
        if pressed_keys: 
            state, command, speedparam, image_mode, video_mode = process_keyboard_command(pressed_keys,state, command, speedparam, image_mode, video_mode)

        #Determine state of drone
        state, command, terminate, image_mode, video_mode, gesture_detected = handle_state(
            state, command, img, speed, perror, gesture_detected, detected_objects, pressed_keys, speedparam, image_mode, video_mode
        )
        if terminate:
            break
        
         # Handle command execution
        vals = handle_command(command, mode, vals, pressed_keys, speed, gesture_detected)
        
        # Handle image and video modes
        image_mode, video_mode = handle_media(image_mode, video_mode, img)

        #Calculate fps
        cTime = time.time()
        if cTime-pTime ==0: fps = 0
        else: fps = 1/(cTime - pTime)
        pTime = cTime
        
        # Update and display parameters
        update_and_display_parameters(
            mode, img, state, command, vals, perror, speedparam, gesture_detected, detected_objects, fps, image_mode, video_mode, battery if mode == "drone" else None
        )
        cv2.imshow("Webcam Object Detection", img)        
        #Stop operation of drone if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    logging.info("Program interrupted by user.")

finally:
    if mode == "drone":
        me.streamoff()
        me.end()
    if mode == "webcam":
        webcam.release()
    cv2.destroyAllWindows()
    logging.info("Drone program terminated.")