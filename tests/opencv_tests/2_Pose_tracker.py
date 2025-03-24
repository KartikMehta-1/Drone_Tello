import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm
import PoseEstimationModule as pem
import numpy as np

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
hand_detector = htm.handDetector()
pose_detector = pem.poseEstimator(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def shoulder_length(leftshoulder,rightshoulder):
    x = rightshoulder[1] - leftshoulder[1]
    y = rightshoulder[2] - leftshoulder[2]
    length = np.sqrt(x*x + y*y)
    return length
    
while True:
    success, img = cap.read()
    #Call the findHands module in handDetector class
    #It returns the image with landmarks and connectors
    img = hand_detector.findHands(img)
    lmList_hand = hand_detector.findPosition(img)
    img = pose_detector.findPoses(img)
    lmList_pose = pose_detector.getPosition(img)
    if len(lmList_pose)!=0:
        if len(lmList_pose[11])!=0 and len(lmList_pose[12])!=0:
            shoulderLength = shoulder_length(lmList_pose[11],lmList_pose[12])
            cv2.circle(img, (lmList_pose[11][1],lmList_pose[11][2]), 20, (255,0,0), cv2.FILLED)
            cv2.circle(img, (lmList_pose[12][1],lmList_pose[12][2]), 20, (255,0,0), cv2.FILLED)
            print(round(shoulderLength,1), lmList_pose[11], lmList_pose[12])
    #Calculate the fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    #Add fps in the video stream
    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow("image", img)
    cv2.waitKey(1)