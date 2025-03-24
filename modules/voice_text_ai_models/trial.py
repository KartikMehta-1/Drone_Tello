import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import noisereduce as nr  # Install using `pip install noisereduce`

# Load Whisper model
model = whisper.load_model("base")

def record_audio_from_microphone(duration=5, fs=16000):
    """
    Record audio from the microphone.
    Args:
        duration (int): Duration of the recording in seconds.
        fs (int): Sampling rate.
    Returns:
        np.ndarray: Recorded audio as a NumPy array.
    """
    print("Recording... Please speak.")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()  # Wait until the recording is finished
    print("Recording completed.")
    return audio.flatten(), fs

def transcribe_audio(audio, fs):
    """
    Transcribe audio using Whisper.
    Args:
        audio (np.ndarray): The audio data.
        fs (int): Sampling rate.
    Returns:
        str: Transcription of the audio.
    """
    # Save audio as temporary file
    temp_filename = "temp_audio.wav"
    write(temp_filename, fs, (audio * 32767).astype(np.int16))  # Convert to 16-bit PCM

    # Load audio and preprocess for Whisper
    result = model.transcribe(temp_filename)
    return result["text"]

# Main Loop
try:
    while True:
        # Step 1: Record audio from the microphone
        audio, fs = record_audio_from_microphone(duration=5, fs=16000)
        
        # Step 2: Optional Noise Reduction
        print("Reducing noise...")
        reduced_audio = nr.reduce_noise(y=audio, sr=fs)

        # Step 3: Transcribe audio with Whisper
        transcription = transcribe_audio(reduced_audio, fs)
        print(f"Transcription: {transcription}")

        # Exit condition
        if transcription.lower() in ["exit", "quit"]:
            print("Exiting program.")
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")



# import speech_recognition as sr
# import pyttsx3
# import whisper
# import numpy as np
# import azure.cognitiveservices.speech as speechsdk

# model = whisper.load_model("base")

# # Initialize Text-to-Speech
# engine = pyttsx3.init()

# def speak(text):
#     """Converts text to speech."""
#     engine.say(text)
#     engine.runAndWait()



# def listen():
#     try:
#         # Initialize Speech SDK
#         speech_config = speechsdk.SpeechConfig(subscription="YourSubscriptionKey", region="YourRegion")
#         speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

#         print("Listening...")
#         result = speech_recognizer.recognize_once()

#         # Check the recognition result
#         if result.reason == speechsdk.ResultReason.RecognizedSpeech:
#             return result.text
#         elif result.reason == speechsdk.ResultReason.NoMatch:
#             return "No speech could be recognized."
#         else:
#             return "An error occurred during speech recognition."
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return "An error occurred while listening."

# def execute_command(command):
#     """Executes simple movement commands based on voice input."""
#     if "move up" in command:
#         return "Command received: Moving up."
#     elif "move down" in command:
#         return "Command received: Moving down."
#     elif "move left" in command:
#         return "Command received: Moving left."
#     elif "move right" in command:
#         return "Command received: Moving right."
#     elif "stop" in command:
#         return "Command received: Stopping."
#     else:
#         return "Command not recognized. Please try again."

# # Main Loop
# try:
#     speak("Hello Kartik, I am ready to receive your commands.")
#     while True:
#         user_input = listen()
#         if user_input.lower() in ["quit", "exit"]:
#             speak("Goodbye, shutting down.")
#             break
#         # Execute Command
#         print(user_input)
#         response = execute_command(user_input.lower())
#         print(response)
#         speak(response)

# except Exception as e:
#     print(f"An error occurred: {e}")

# def listen():
#     """Captures voice input."""
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         print("Listening...")
#         audio = recognizer.listen(source,  timeout=5, phrase_time_limit=10)
#     try:
#         return recognizer.recognize_google(audio)
#     except sr.UnknownValueError:
#         return "Sorry, I didn't understand that."

# def listen():
#     """Captures voice input and transcribes using Whisper."""
#     recognizer = sr.Recognizer()
#     with sr.Microphone(sample_rate=16000) as source:
#         print("Listening...")
#         audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

#     try:
#         # Convert audio to NumPy array
#         wav_data = audio.get_wav_data()
#         audio_data = np.frombuffer(wav_data, np.int16).astype(np.float32) / 32768.0  # Normalize to [-1, 1]

#         # Run transcription
#         result = model.transcribe(audio_data)
#         return result['text']
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return "Sorry, I didn't catch that."