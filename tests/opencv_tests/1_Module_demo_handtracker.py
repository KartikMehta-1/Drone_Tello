import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()
while True:
    success, img = cap.read()
    #Call the findHands module in handDetector class
    #It returns the image with landmarks and connectors
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) !=0: print(lmList[4])
    #Calculate the fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    #Add fps in the video stream
    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow("image", img)
    cv2.waitKey(1)
