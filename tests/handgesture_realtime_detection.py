import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import joblib
from collections import deque
import sys
sys.path.append('D:\Kartik\learning\AI_Projects\OpenCV_Tutorials')
import HandTrackingModule as htm

hand_detector = htm.handDetector(detConf=0.7, trckConf=0.7)
# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Load the trained model
model = load_model("hand_gesture_model.h5")

# Load the label encoder used during training
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("gesture_classes.npy",allow_pickle=True)  # Save and load the classes from training

# Initialize scaler
scaler = joblib.load("scaler.pkl")

while True:
    success, img = cap.read()
    if not success:
        break
    lmList_long = hand_detector.findPosition_long(img)
    if lmList_long: 
        processed_lmList = hand_detector.preprocess_lmList(lmList_long)
        flat_landmarks_array = np.array(processed_lmList).reshape(1, -1)  # Convert to a numpy of shape to (1, 42)
        prediction = model.predict(flat_landmarks_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = prediction[0][predicted_class_index]
        # Display the predicted gesture on the frame
        text = f"Gesture: {predicted_class} ({confidence:.2f})"
        cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
  
    # Keyboard controls - Quit operation
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
    #Display image
    cv2.imshow("Hand Gesture Capture", img)
 
# Release resources
cap.release()
cv2.destroyAllWindows()