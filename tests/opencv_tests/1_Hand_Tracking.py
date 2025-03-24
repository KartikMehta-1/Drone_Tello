import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands #Formality steps before starting to use model
hands = mpHands.Hands() # Hand detection model by mediapipe - Parameters: Static mode = false, max. number of hands, min confidence of detection & tracking
mpDraw = mp.solutions.drawing_utils # Mediapipe module to draw landmarks & connectors on display

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print all points of hands their ids, and coordinates
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id,cx,cy)
                # Showing how to highlight any of the point.
                if id == 0:
                    cv2.circle(img, (cx,cy), 25, (255,0,255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    #Add fps in the video stream
    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv2.imshow("image", img)
    cv2.waitKey(1)