import cv2
import mediapipe as mp
import time

class poseEstimator():
    def __init__(self, mode=False,
               model_complexity=1,
               smooth_landmarks=True,
               enable_segmentation=False,
               smooth_segmentation=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        #Formality steps before starting to use model
        self.mpPose = mp.solutions.pose
        #mediapipe function to return the identified poses
        self.pose = self.mpPose.Pose(
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        #mediapipe function to access drawing utilities in mediapipe
        self.mpDraw = mp.solutions.drawing_utils

    def findPoses(self,img,draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks: 
            if draw: self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    
    def getPosition(self,img,draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                    h,w,c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    lmList.append([id,cx,cy])
                    if draw: cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED)
        return lmList
    

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    poseEstimate = poseEstimator()
    while True:
        success, img = cap.read()
        img = poseEstimate.findPoses(img)
        lmList = poseEstimate.getPosition(img)
        if len(lmList) !=0: print(lmList[4])
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