import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
data = pd.read_csv("gesture_data/hand_gestures.csv")

# Prepare input (X) and output (y)
X = data.drop(columns=["gesture_name"]).values
Y = data["gesture_name"].values
# # Ensure y is a flattened 1D array

# Encode gesture labels into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(Y)
print(y_encoded)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)
# One-hot encode labels for training with categorical_crossentropy
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input((X_train.shape[1], )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
])

print(model.summary()) 

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=128,
    validation_data=(X_test, y_test),
    #callbacks=[cp_callback, es_callback]
)


# # Standardize the features to have zero mean and unit variance
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)


# print("post X-scaling (zero mean and unit variance) and Y - categorical")
# print("X_train: ", X_train.shape)
# print("y_train: ", y_train.shape)
# print("X_test: ", X_test.shape)
# print("y_test: ", y_test.shape)

#Define a simplified neural network model
# model = Sequential([
#      Dense(256, input_shape=(X_train.shape[1],), activation="relu"),
#      BatchNormalization(),
#      Dropout(0.5),
#      Dense(128, activation="relu"),
#      BatchNormalization(),
#      Dropout(0.4),
#      Dense(64, activation="relu"),
#      BatchNormalization(),
#      Dropout(0.3),
#      Dense(32, activation="relu"),
#      BatchNormalization(),
#      Dropout(0.2),
#      Dense(y_train.shape[1], activation="softmax")
#  ])




# Set up a learning rate scheduler
# lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1)


# Train the model
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16)

#Model checkpoint callback
# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     model_save_path, verbose=1, save_weights_only=False)
# # Callback for early stopping
# es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)



# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Print the classification report and accuracy
print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_))

print("Accuracy Score:", accuracy_score(y_true_classes, y_pred_classes))

# Save the trained model
model.save("hand_gesture_model.h5")
print("Model saved as hand_gesture_model.h5")

# Save the label encoder classes
np.save("gesture_classes.npy", label_encoder.classes_)
print("Label encoder classes saved as gesture_classes.npy")

# # Save the scaler parameters
# joblib.dump(scaler, "scaler.pkl")
# print("Scaler saved as scaler.pkl")