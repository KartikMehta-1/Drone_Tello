import cv2
import mediapipe as mp
import time

class holisticEstimator():
    def __init__(self,
               static_image_mode=False,
               model_complexity=1,
               smooth_landmarks=True,
               enable_segmentation=False,
               smooth_segmentation=True,
               refine_face_landmarks=False,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.refine_face_landmarks = refine_face_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mpHolistic = mp.solutions.holistic
        self.holistic = self.mpHolistic.Holistic(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            refine_face_landmarks = self.refine_face_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        #mediapipe function to access drawing utilities in mediapipe
        self.mpDraw = mp.solutions.drawing_utils
    
    def drawHolistic(self,img,poseDraw = True, handDraw = True, faceDraw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.holistic.process(imgRGB)
        if poseDraw:
            if self.results.pose_landmarks: 
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpHolistic.POSE_CONNECTIONS)
        if handDraw:
            if self.results.left_hand_landmarks: 
                self.mpDraw.draw_landmarks(img, self.results.left_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS)
            if self.results.right_hand_landmarks: 
                self.mpDraw.draw_landmarks(img, self.results.right_hand_landmarks, self.mpHolistic.HAND_CONNECTIONS)
        if faceDraw:
            if self.results.face_landmarks: 
                self.mpDraw.draw_landmarks(img, self.results.face_landmarks, self.mpHolistic.FACEMESH_TESSELATION)
        return img
    
    def getPosePosition(self,img,draw = True):
        lmList = []
        lmListReturn = []
        if self.results.pose_landmarks:
            lmList = [landmark for landmark in self.results.pose_landmarks.landmark]
            for index, landmark in enumerate(lmList):
                h,w,c = img.shape
                cx, cy = int(landmark.x*w), int(landmark.y*h)
                lmListReturn.append([index,cx,cy])
        return lmListReturn
        
    def getFacePosition(self,img,draw = True):
        lmList = []
        lmListReturn = []
        if self.results.face_landmarks:
            lmList = [landmark for landmark in self.results.face_landmarks.landmark]
            for index, landmark in enumerate(lmList):
                h,w,c = img.shape
                cx, cy = int(landmark.x*w), int(landmark.y*h)
                lmListReturn.append([index,cx,cy])
        return lmListReturn
    
    def getLeftHandPosition(self,img,draw = True):
        lmList = []
        lmListReturn = []
        if self.results.left_hand_landmarks:
            lmList = [landmark for landmark in self.results.left_hand_landmarks.landmark]
            for index, landmark in enumerate(lmList):
                h,w,c = img.shape
                cx, cy = int(landmark.x*w), int(landmark.y*h)
                lmListReturn.append([index,cx,cy])
        return lmListReturn
    
    def getRightHandPosition(self,img,draw = True):
        lmList = []
        lmListReturn = []
        if self.results.right_hand_landmarks:
            lmList = [landmark for landmark in self.results.right_hand_landmarks.landmark]
            for index, landmark in enumerate(lmList):
                h,w,c = img.shape
                cx, cy = int(landmark.x*w), int(landmark.y*h)
                lmListReturn.append([index,cx,cy])
        return lmListReturn
        

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    holisticEstimate = holisticEstimator()
    while True:
        success, img = cap.read()
        img = holisticEstimate.drawHolistic(img, faceDraw=False, poseDraw=False)
        lmListpose = holisticEstimate.getPosePosition(img)
        lmListface = holisticEstimate.getFacePosition(img)
        lmListLHand = holisticEstimate.getLeftHandPosition(img)
        lmListRHand = holisticEstimate.getRightHandPosition(img)
        if len(lmListpose) !=0: print(lmListpose[4])
        #Calculate the fps
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        #Add fps in the video stream
        cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow("image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()