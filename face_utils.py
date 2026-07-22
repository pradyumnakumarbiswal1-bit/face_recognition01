import os
import pickle
import face_recognition
import numpy as np
from pathlib import Path


class FaceRecognitionSystem:
    def __init__(self, known_faces_dir='known_faces', encodings_file='face_encodings.pkl'):
        self.known_faces_dir = known_faces_dir
        self.encodings_file = encodings_file
        self.known_encodings = []
        self.known_names = []
        os.makedirs(known_faces_dir, exist_ok=True)
        self.load_encodings()
    
    def load_encodings(self):
        if os.path.exists(self.encodings_file):
            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_encodings = data['encodings']
                self.known_names = data['names']
            print(f"Loaded {len(self.known_encodings)} face encodings")
        else:
            if self.get_registered_people():
                print("No existing encodings found. Generating from known_faces folder...")
                self.generate_encodings_from_folder()
            else:
                print("No existing encodings found. Register faces first.")
    
    def save_encodings(self):
        data = {
            'encodings': self.known_encodings,
            'names': self.known_names
        }
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {len(self.known_encodings)} face encodings")
    
    def generate_encodings_from_folder(self):
        self.known_encodings = []
        self.known_names = []
        
        for person_name in os.listdir(self.known_faces_dir):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            
            if not os.path.isdir(person_dir):
                continue
            
            print(f"Processing {person_name}...")
            
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if len(face_encodings) > 0:
                        encoding = face_encodings[0]
                        self.known_encodings.append(encoding)
                        self.known_names.append(person_name)
                        print(f"  ✓ Encoded {image_name}")
                    else:
                        print(f"  ✗ No face detected in {image_name}")
                
                except Exception as e:
                    print(f"  ✗ Error processing {image_name}: {str(e)}")
        
        self.save_encodings()
        print(f"Total encodings generated: {len(self.known_encodings)}")
    
    def recognize_face(self, face_encoding, tolerance=0.6):
        if len(self.known_encodings) == 0:
            return "Unknown", 0.0
        matches = face_recognition.compare_faces(
            self.known_encodings, 
            face_encoding, 
            tolerance=tolerance
        )
        face_distances = face_recognition.face_distance(
            self.known_encodings, 
            face_encoding
        )
        if len(face_distances) == 0:
            return "Unknown", 0.0
        best_match_idx = np.argmin(face_distances)
        best_distance = face_distances[best_match_idx]
        if matches[best_match_idx]:
            confidence = 1 - best_distance
            name = self.known_names[best_match_idx]
            return name, confidence
        else:
            return "Unknown", 0.0
    
    def get_registered_people(self):
        people = []
        for person_name in os.listdir(self.known_faces_dir):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            if os.path.isdir(person_dir):
                image_count = len([f for f in os.listdir(person_dir) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                people.append((person_name, image_count))
        return people


def draw_face_box_with_label(frame, top, right, bottom, left, name, confidence=None):
    color = (0, 255, 0)
    if name == "Unknown":
        color = (0, 0, 255)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    label = name
    if confidence is not None:
        label = f"{name} ({confidence:.2f})"
    label_y = top - 10 if top > 30 else bottom + 25
    cv2.putText(frame, label, (left, label_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
try:
    import cv2
except ImportError:
    print("OpenCV not installed. Install with: pip install opencv-python")
