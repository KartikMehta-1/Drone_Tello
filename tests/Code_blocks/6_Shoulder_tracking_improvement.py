import cv2
import numpy as np
from djitellopy import tello
from time import sleep
import time
import sys
sys.path.append('D:\Kartik\learning\AI_Projects\OpenCV_Tutorials')
import PoseEstimationModule as pem

me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()
me.takeoff()
me.send_rc_control(0,0,25,0)
#time.sleep(5.0)
w,h = 360,240
shLenRange = [50,100]
pid = [0.6,0.4,0]
pid_height = [0.6,0.4,0]
pError = 0
pError_height = 0
pose_detector = pem.poseEstimator(min_detection_confidence=0.7, min_tracking_confidence=0.7)
pTime = 0
cTime = 0

def shoulder_length(leftshoulder,rightshoulder):
    x = rightshoulder[1] - leftshoulder[1]
    y = rightshoulder[2] - leftshoulder[2]
    cx = (rightshoulder[1] + leftshoulder[1])/2
    cy = (rightshoulder[2] + leftshoulder[2])/2
    length = np.sqrt(x*x + y*y)
    print(length)
    return (length,cx,cy)

def shoulder_head(leftshoulder,head):
    x = head[1] - leftshoulder[1]
    y = head[2] - leftshoulder[2]
    length = np.sqrt(x*x + y*y)
    print(length)
    return (length)

def trackPose(me, shoulderLength, shoulderHead,cx, cy, w, h, pid, pError, pid_height, pError_height):
    error = cx-w//2
    speed = pid[0] * error + pid[1]* (error -pError)
    speed = int(np.clip(speed,-100,100))
    error_height = cy - h//2
    speed_height = pid_height[0] * error_height + pid_height[1] * (error_height - pError_height)
    speed_height = int(np.clip(speed_height,-100,100))
    if shoulderHead >shLenRange[0] and shoulderHead < shLenRange[1]:
        fb = 0
    elif shoulderHead >shLenRange[1]:
        fb = -20
    #elif shoulderLength < shLenRange[0] and shoulderLength!= 0:
    elif shoulderHead < shLenRange[0]:
        fb = 20
    else: fb = 0

    if cx == 0:
        speed = 0
        error = 0
        speed_height = 0
    me.send_rc_control(0,fb,-speed_height,speed)
    return (error, error_height)

while True:
    #_, img = cap.read()
    img = me.get_frame_read().frame #get frame from drone
<<<<<<< HEAD
=======
    img = cv2.resize(img,(w,h))
>>>>>>> 0e0fe32dc5fb2aafb587c30aa823534cb0319bf1
    img = pose_detector.findPoses(img)
    lmList_pose = pose_detector.getPosition(img)
    if len(lmList_pose)!=0:
        if len(lmList_pose[11])!=0 and len(lmList_pose[12])!=0:
            shoulderLength, cx, cy = shoulder_length(lmList_pose[11],lmList_pose[12])
            shoulderHead = shoulder_head(lmList_pose[11],lmList_pose[3])
            pError,pError_height = trackPose(me, shoulderLength, shoulderHead, cx, cy, w, h, pid, pError, pid_height, pError_height)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow("Output",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break