import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHolistic = mp.solutions.holistic #Formality steps before starting to use model
holistic = mpHolistic.Holistic() # Hand detection model by mediapipe - Parameters: Static mode = false, max. number of hands, min confidence of detection & tracking
mpDraw = mp.solutions.drawing_utils # Mediapipe module to draw landmarks & connectors on display

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = holistic.process(imgRGB)
    #Drawing all landmarks
    if results.left_hand_landmarks: mpDraw.draw_landmarks(img, results.left_hand_landmarks, mpHolistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks: mpDraw.draw_landmarks(img, results.right_hand_landmarks, mpHolistic.HAND_CONNECTIONS)
    if results.face_landmarks: mpDraw.draw_landmarks(img, results.face_landmarks, mpHolistic.FACEMESH_TESSELATION)
    if results.pose_landmarks: mpDraw.draw_landmarks(img, results.pose_landmarks, mpHolistic.POSE_CONNECTIONS)
    #Printing nose from pose landmark
    if results.pose_landmarks:
        lmList = [landmark for landmark in results.pose_landmarks.landmark]
        for index, landmark in enumerate(lmList):
            print(f"Landmark {index}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
            h,w,c = img.shape
            cx, cy = int(landmark.x*w), int(landmark.y*h)
            if index == 0:
                    cv2.circle(img, (cx,cy), 25, (255,0,255), cv2.FILLED)
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    #Add fps in the video stream
    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv2.imshow("image", img)
    cv2.waitKey(1)