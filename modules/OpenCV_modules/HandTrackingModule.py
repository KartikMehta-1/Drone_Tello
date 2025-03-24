import cv2
import mediapipe as mp
import time
import numpy as np
import itertools

class handDetector():
    def __init__(self, mode = False, maxHands=1, detConf = 0.5, trckConf = 0.5):
        self.mode = mode #Create an object, which will have a variable called mode.
        self.maxHands = maxHands
        self.detConf = detConf
        self.trckConf = trckConf
        self.mpHands = mp.solutions.hands #Formality steps before starting to use model
        self.hands = self.mpHands.Hands(
            static_image_mode = self.mode,
            max_num_hands = self.maxHands,
            min_detection_confidence = self.detConf,
            min_tracking_confidence = self.trckConf) # Hand detection model by mediapipe - Parameters: Static mode = false, max. number of hands, min confidence of detection & tracking
        self.mpDraw = mp.solutions.drawing_utils # Mediapipe module to draw landmarks & connectors on display
    
    #Returns the image with landmarks
    def findHands(self,img, draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw: self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    
    #Returns the list of ids and coordinates of landmarks. 
    def findPosition(self, img, handNo=0, draw = True):
        lmList = []
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([cx,cy])
                if draw: cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
        return lmList
    
    #Returns a single extended list of coordinates of landmarks.
    def findPosition_long(self, img, handNo=0, draw = True):
        lmList = []
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        landmark_array = np.empty((0, 2), int)
        x=0
        y=0
        l=0
        b=0
        h,w,c = img.shape
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.extend([cx,cy])
                landmark_x = min(int(lm.x * w), w - 1)
                landmark_y = min(int(lm.y * h), h - 1)
                landmark_point = [np.array((landmark_x, landmark_y))]
                landmark_array = np.append(landmark_array, landmark_point, axis=0)
                x, y, l, b = cv2.boundingRect(landmark_array)
                if draw: 
                    cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
            if draw: cv2.rectangle(img, (x, y), (x+l, y+b), (0, 255, 0), 2)
        return lmList
    
    #Returns the coordinates of bounding box of hand in image.
    def find_bbox(self, img, handNo=0, draw=True):
        w, h = img.shape[1], img.shape[0]
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        landmark_array = np.empty((0, 2), int)
        x=0
        y=0
        l=0
        b=0
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                landmark_x = min(int(lm.x * w), w - 1)
                landmark_y = min(int(lm.y * h), h - 1)
                landmark_point = [np.array((landmark_x, landmark_y))]
                landmark_array = np.append(landmark_array, landmark_point, axis=0)
                x, y, l, b = cv2.boundingRect(landmark_array)
            if draw: cv2.rectangle(img, (x, y), (x+l, y+b), (0, 255, 0), 2)
        return [x, y, x + l, y + b]
    
    def find_rel_posn_bbox(self,bbox, img, handNo=0, draw=True):
        xmin = bbox[0]
        ymin = bbox[1]
        image_width, image_height = img.shape[1], img.shape[0]
        rel_landmarks = []
        rel_landmarks_normalized = []
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            landmarks = self.results.multi_hand_landmarks[handNo]
            for _, landmark in enumerate(landmarks.landmark):
                rel_landmarks.extend([int(landmark.x * image_width) - xmin,int(landmark.y * image_height) - ymin])
            max_value = max(rel_landmarks)
            rel_landmarks_normalized = [x/max_value for x in rel_landmarks]
        return rel_landmarks_normalized
    
    def preprocess_lmList(self,lmList_long):
        xmin = lmList_long[0]
        ymin = lmList_long[1]
        for i in range(len(lmList_long)):
            if i%2 == 0:
                lmList_long[i] = lmList_long[i]-xmin
            else: 
                lmList_long[i] = lmList_long[i] - ymin
        
        max_value = max(list(map(abs, lmList_long)))

        def normalize_(n):
            return n / max_value
        temp_landmark_list = list(map(normalize_, lmList_long))
            
        return temp_landmark_list
        


# if name == main will make sure that the scripts in this area is only executed when this file is run directly.
# If this file is imported in another function, then the following is not run, but other functions or classes can be used
#Dummy main function to show what this module file can do.
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        lmList_long = detector.findPosition_long(img)
        #bbox = detector.find_bbox(img)
      
        #Calculate the fps
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        #Add fps in the video stream
        cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow("image", img)
        #cv2.waitKey(1)
           
        # Keyboard controls - Quit operation
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

if __name__ == "__main__":
    main()