# import tensorflow as tf

# print("TensorFlow version:", tf.__version__)
# print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
# import torch

# def check_gpu():
#     if torch.cuda.is_available():
#         print("✅ GPU is available!")
#         print(f"GPU Name: {torch.cuda.get_device_name(0)}")
#         print(f"CUDA Version: {torch.version.cuda}")
#     else:
#         print("❌ GPU is not available. Using CPU.")

# if __name__ == "__main__":
#     check_gpu()

# print("hello")

import pyttsx3
import json
from djitellopy import tello
import speech_recognition as sr
print("Basic imports successful")

drone = tello.Tello()
print("Tello initialized")
