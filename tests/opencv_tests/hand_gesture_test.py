import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

model = load_model("hand_gesture_model.h5")

# Load the label encoder used during training
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("gesture_classes.npy",allow_pickle=True)  # Save and load the classes from training

data = pd.read_csv("gesture_data/hand_gestures.csv")
X = data.drop(columns=["gesture_name"]).values
row = 3500

flat_landmarks_array = X[row].reshape(1, -1)  # Convert to a numpy of shape to (1, 42)
prediction = model.predict(flat_landmarks_array)
predicted_class_index = np.argmax(prediction)
predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
confidence = prediction[0][predicted_class_index]
# Display the predicted gesture on the frame
text = f"Gesture: {predicted_class} ({confidence:.2f})"

print("Prediction: ",prediction, "Class index: ", predicted_class_index)
print("expected: ",data.iloc[row][0], "predicted: ",text)
