import os
import cv2
import mediapipe as mp
import json
from deepface import DeepFace
import numpy as np

import hashlib

class FaceRecognition:
    def __init__(self, known_faces_folder, embeddings_file="embeddings.json", model="Facenet", detection_confidence=0.5, recognition_threshold=0.4):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=detection_confidence
        )
        self.model = model
        self.recognition_threshold = recognition_threshold
        self.embeddings_file = embeddings_file
        self.known_faces_folder = known_faces_folder

        # Check for changes in the known faces folder
        if self._is_folder_changed(known_faces_folder, embeddings_file):
            print("Changes detected in the folder. Recalculating embeddings...")
            self.face_embeddings = self._preprocess_known_faces(known_faces_folder)
            self._save_embeddings(self.face_embeddings, embeddings_file)
        else:
            print("No changes detected. Loading embeddings from file...")
            self.face_embeddings = self._load_embeddings(embeddings_file)

    def _get_folder_hash(self, folder_path):
        """Generate a hash based on folder structure and file modification times."""
        hash_md5 = hashlib.md5()
        for root, _, files in os.walk(folder_path):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                hash_md5.update(file_path.encode())  # Hash file path
                hash_md5.update(str(os.path.getmtime(file_path)).encode())  # Hash last modification time
        return hash_md5.hexdigest()

    def _is_folder_changed(self, folder_path, embeddings_file):
        """Check if the folder content has changed by comparing hashes."""
        folder_hash = self._get_folder_hash(folder_path)
        if os.path.exists(embeddings_file):
            with open(embeddings_file, "r") as f:
                data = json.load(f)
            return data.get("folder_hash") != folder_hash
        return True

    def _preprocess_known_faces(self, known_faces_folder):
        """Calculate embeddings for known faces."""
        face_embeddings = {}
        for person_name in os.listdir(known_faces_folder):
            person_folder = os.path.join(known_faces_folder, person_name)
            if os.path.isdir(person_folder):
                person_embeddings = []
                for image_file in os.listdir(person_folder):
                    image_path = os.path.join(person_folder, image_file)
                    try:
                        embedding = DeepFace.represent(img_path=image_path, model_name=self.model)
                        person_embeddings.append(embedding[0]["embedding"])
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
                
                if person_embeddings:
                    # Average embeddings for the person
                    face_embeddings[person_name] = np.mean(person_embeddings, axis=0).tolist()  # Convert to list for JSON serialization
        return face_embeddings

    def _save_embeddings(self, embeddings, file_path):
        """Save embeddings and folder hash to a file."""
        folder_hash = self._get_folder_hash(self.known_faces_folder)
        data = {"folder_hash": folder_hash, "embeddings": embeddings}
        with open(file_path, "w") as f:
            json.dump(data, f)
        print(f"Embeddings and folder hash saved to {file_path}")

    def _load_embeddings(self, file_path):
        """Load embeddings from a file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return {name: np.array(embedding) for name, embedding in data["embeddings"].items()}  # Convert back to numpy arrays

    @staticmethod
    def cosine_distance(vector1, vector2):
        """Calculate cosine distance between two vectors."""
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        return 1 - (dot_product / (norm1 * norm2))

    def identify_face(self, face_image_path):
        """Identify a face by comparing its embedding with known embeddings."""
        try:
            target_embedding = DeepFace.represent(img_path=face_image_path, model_name=self.model)[0]["embedding"]
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return "Unknown"
        
        for name, embedding in self.face_embeddings.items():
            distance = self.cosine_distance(target_embedding, embedding)
            if distance < self.recognition_threshold:
                return name
        return "Unknown"

    def process_frame(self, frame):
        """Detect and identify faces in the given frame."""
        results = self.face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        identities = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Extract face region
                face_image = frame[y:y+h, x:x+w]
                face_image_path = "temp_face.jpg"
                cv2.imwrite(face_image_path, face_image)

                # Identify face
                identity = self.identify_face(face_image_path)
                identities.append((identity, (x, y, w, h)))
        return identities


if __name__ == "__main__":
    # Path to known faces folder and embeddings file
    KNOWN_FACES_FOLDER = "D:/Kartik/learning/AI_Projects/face_recognition/Known_Faces"  # Replace with your folder path
    EMBEDDINGS_FILE = "embeddings.json"

    # Initialize the face recognition system
    face_recognizer = FaceRecognition(known_faces_folder=KNOWN_FACES_FOLDER, embeddings_file=EMBEDDINGS_FILE)

    # Open webcam or video stream
    cap = cv2.VideoCapture(0)
    print("Starting face detection and recognition...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Process the frame for detection and recognition
        identities = face_recognizer.process_frame(frame)

        # Draw results on the frame
        for identity, (x, y, w, h) in identities:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, identity, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Display the video feed
        cv2.imshow('Face Detection and Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
