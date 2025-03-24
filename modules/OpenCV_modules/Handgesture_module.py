import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append('D:\Kartik\learning\AI_Projects\OpenCV_Tutorials')
import HandTrackingModule as htm
import time

hand_detector = htm.handDetector(detConf=0.7, trckConf=0.7)

class HandGesture:
    def __init__(self,model_path,label_path,gesture_confidence=0.5):
        self.model = load_model(model_path)
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.load(label_path,allow_pickle=True)  # Save and load the classes from training
        self.confidence = gesture_confidence

    def detect_gesture(self,frame):
        lmList_long = hand_detector.findPosition_long(frame)
        if lmList_long: 
            processed_lmList = hand_detector.preprocess_lmList(lmList_long)
            flat_landmarks_array = np.array(processed_lmList).reshape(1, -1)  # Convert to a numpy of shape to (1, 42)
            prediction = self.model.predict(flat_landmarks_array)
            predicted_class_index = np.argmax(prediction)
            predicted_class = self.label_encoder.inverse_transform([np.argmax(prediction)])[0]
            confidence = prediction[0][predicted_class_index]
            return(predicted_class,confidence)

def main():
    # parser = argparse.ArgumentParser(description="Handgesture Detection")
    # parser.add_argument("--model", required=True, help="Path to model")
    # parser.add_argument("--label", required=True, help="Path to label")
    # parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    # args = parser.parse_args()
    # model_trained = HandGesture(args.model, args.label, args.confidence)
    model_path = "D:\Kartik\Projects\Drone_Tello\models\hand_gesture_trained_model\hand_gesture_model.h5"
    label_path = "D:\Kartik\Projects\Drone_Tello\models\hand_gesture_trained_model\gesture_classes.npy"
    
    model_trained = HandGesture(model_path, label_path, 0.5)
    # Open the webcam
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    if not cap.isOpened():
        print("Error: Unable to access webcam.")
        return

    print("Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break

            # Detect objects
            detections = model_trained.detect_gesture(frame)

            # Display the frame
            
            if detections: print(detections)
            cTime = time.time()
            fps = 1/(cTime - pTime)
            pTime = cTime
                
            cv2.putText(frame, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
            
            if detections: cv2.putText(frame, str(detections[0]),(10,30),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

            cv2.imshow("Webcam Object Detection", frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()