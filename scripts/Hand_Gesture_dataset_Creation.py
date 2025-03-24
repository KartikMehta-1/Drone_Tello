import cv2
import numpy as np
import pandas as pd
from time import sleep
import time
import sys
import os
sys.path.append('D:\Kartik\learning\AI_Projects\OpenCV_Tutorials')
import HandTrackingModule as htm
from itertools import chain

hand_detector = htm.handDetector(detConf=0.7, trckConf=0.7)
cap = cv2.VideoCapture(0)
classes = ["GoAway","ComeNear","Up","Down","Left","Right","Spin","Flip"]
data = []
print("Press 's' to start capturing gestures, 'd' to stop and 'q' to quit.")
gesture_id = -1
output_dir = "gesture_data"
os.makedirs(output_dir, exist_ok=True)
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    if not success:
        break
    lmList_long = hand_detector.findPosition_long(img)
    if gesture_id >= 0:  # Only save if a gesture has been selected
        if lmList_long: 
            processed_lmList = hand_detector.preprocess_lmList(lmList_long)
            processed_lmList.insert(0, classes[gesture_id])  # Insert the label at the beginning
            data.append(processed_lmList)
            cv2.putText(img, f"Recording gesture: {classes[gesture_id]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #Calculate the fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    #Add fps in the video stream
    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    
    #Display image
    cv2.imshow("Hand Gesture Capture", img)

    # Keyboard controls - Quit operation
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # Capture the frames for training
    elif key == ord('s'):  # Start capturing for a selected gesture
        print("Press the number key for the gesture you want to capture:")
        for idx, gesture in enumerate(classes):
            print(f"{idx}: {gesture}")
    elif key == ord('d'):  # Stop capturing for a selected gesture
        gesture_id = -1
    elif key >= ord('0') and key <= ord(str(len(classes) - 1)):
        gesture_id = int(chr(key))
        print(f"Capturing {classes[gesture_id]} gesture")

# Release resources
cap.release()
cv2.destroyAllWindows()
# Save data to CSV
columns = ['gesture_name'] + [f'x{i//2+1}' if i % 2 == 0 else f'y{i//2+1}' for i in range(21 * 2)]
df = pd.DataFrame(data, columns=columns)
<<<<<<< HEAD

# Write to CSV in append mode
#df.to_csv(csv_file, mode='a', header=not file_exists, index=False)
#print(f"Dataset appended to {csv_file}")

df.to_csv(os.path.join(output_dir, "hand_gestures_1.csv"), index=False)
print("Dataset saved as hand_gestures.csv_1")
=======
df.to_csv(os.path.join(output_dir, "hand_gestures.csv"), index=False)
print("Dataset saved as hand_gestures.csv")
>>>>>>> 0e0fe32dc5fb2aafb587c30aa823534cb0319bf1
