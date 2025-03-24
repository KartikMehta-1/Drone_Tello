import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm
import PoseEstimationModule as pem

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
hand_detector = htm.handDetector()
pose_detector = pem.poseEstimator()
while True:
    success, img = cap.read()
    #Call the findHands module in handDetector class
    #It returns the image with landmarks and connectors
    img = hand_detector.findHands(img)
    lmList_hand = hand_detector.findPosition(img)
    img = pose_detector.findPoses(img)
    lmList_pose = pose_detector.getPosition(img)
    if len(lmList_pose) !=0: print(lmList_pose[0])
    #Calculate the fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    #Add fps in the video stream
    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow("image", img)
    cv2.waitKey(1)
