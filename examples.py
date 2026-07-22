"""
Example usage of the Face Recognition System
Run this file to see demonstrations of different features
"""
import cv2
import face_recognition
import os
from face_utils import FaceRecognitionSystem
from face_recognition_real_time import RealTimeFaceRecognition


def example_1_register_people():
    """Example 1: Register multiple people"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Register Multiple People")
    print("="*70)
    print("""
This example shows how to register new people programmatically.

Code:
    system = FaceRecognitionSystem()
    system.generate_encodings_from_folder()
    people = system.get_registered_people()
    print(people)
    """)


def example_2_recognize_face():
    """Example 2: Recognize a single face"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Recognize a Single Face")
    print("="*70)
    print("""
This example shows how to recognize a face in an image.

Code:
    import face_recognition
    from face_utils import FaceRecognitionSystem
    
    system = FaceRecognitionSystem()
    
    # Load image and get face encoding
    image = face_recognition.load_image_file("unknown_face.jpg")
    face_encodings = face_recognition.face_encodings(image)
    
    if len(face_encodings) > 0:
        face_encoding = face_encodings[0]
        name, confidence = system.recognize_face(face_encoding)
        print(f"{name} (confidence: {confidence:.2f})")
    """)


def example_3_real_time_webcam():
    """Example 3: Real-time webcam recognition"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Real-Time Webcam Recognition")
    print("="*70)
    print("""
This example shows real-time face recognition from webcam.

Code:
    from face_recognition_real_time import RealTimeFaceRecognition
    
    recognizer = RealTimeFaceRecognition(model='hog')
    recognizer.run_webcam(tolerance=0.6, show_confidence=True)
    
Controls during execution:
    - 'q' to quit
    - 's' to save current frame
    - 'p' to print detected faces
    """)


def example_4_process_image():
    """Example 4: Process an image file"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Process Image File")
    print("="*70)
    print("""
This example shows how to process a single image.

Code:
    from face_recognition_real_time import RealTimeFaceRecognition
    
    recognizer = RealTimeFaceRecognition()
    recognizer.process_image(
        'input_image.jpg',
        tolerance=0.6,
        output_path='output_image.jpg'
    )
    """)


def example_5_process_video():
    """Example 5: Process a video file"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Process Video File")
    print("="*70)
    print("""
This example shows how to process a video file.

Code:
    from face_recognition_real_time import RealTimeFaceRecognition
    
    recognizer = RealTimeFaceRecognition()
    recognizer.process_video(
        'input_video.mp4',
        tolerance=0.6,
        output_path='output_video.mp4'
    )
    """)


def example_6_batch_processing():
    """Example 6: Batch processing multiple images"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Batch Processing Multiple Images")
    print("="*70)
    print("""
This example shows how to process multiple images in batch.

Code:
    import os
    from face_recognition_real_time import RealTimeFaceRecognition
    
    recognizer = RealTimeFaceRecognition()
    
    input_dir = 'images_to_process'
    output_dir = 'processed_images'
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            recognizer.process_image(input_path, output_path=output_path)
    """)


def example_7_custom_tolerance():
    """Example 7: Custom tolerance for different use cases"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Custom Tolerance Settings")
    print("="*70)
    print("""
Different tolerance values for different scenarios:

1. STRICT (High Security):
   - Tolerance: 0.4
   - Use case: Access control, security verification
   - Only accepts very confident matches
   
Code:
    recognizer = RealTimeFaceRecognition()
    recognizer.run_webcam(tolerance=0.4)

2. BALANCED (Default):
   - Tolerance: 0.6
   - Use case: General face recognition
   - Good accuracy with reasonable flexibility
   
Code:
    recognizer = RealTimeFaceRecognition()
    recognizer.run_webcam(tolerance=0.6)

3. LENIENT (High Sensitivity):
   - Tolerance: 0.8
   - Use case: Finding similar faces, duplicate detection
   - More matches but higher false positive rate
   
Code:
    recognizer = RealTimeFaceRecognition()
    recognizer.run_webcam(tolerance=0.8)
    """)


def example_8_model_selection():
    """Example 8: Choosing between HOG and CNN models"""
    print("\n" + "="*70)
    print("EXAMPLE 8: Model Selection (HOG vs CNN)")
    print("="*70)
    print("""
Two detection models available:

1. HOG (Histogram of Oriented Gradients):
   - Speed: FAST (~30 FPS)
   - Accuracy: GOOD (95%)
   - CPU: Low usage
   - Best for: Real-time webcam
   
Code:
    recognizer = RealTimeFaceRecognition(model='hog')
    recognizer.run_webcam()

2. CNN (Convolutional Neural Network):
   - Speed: SLOW (~5 FPS)
   - Accuracy: EXCELLENT (99%)
   - CPU: High usage
   - Best for: Static images, high accuracy needed
   
Code:
    recognizer = RealTimeFaceRecognition(model='cnn')
    recognizer.process_image('photo.jpg')

Recommendation:
    - Use HOG for webcam/live video
    - Use CNN for still images or when accuracy is critical
    """)


def example_9_advanced_filtering():
    """Example 9: Advanced face filtering and analysis"""
    print("\n" + "="*70)
    print("EXAMPLE 9: Advanced Filtering and Analysis")
    print("="*70)
    print("""
Example: Filter faces by confidence and track detections

Code:
    import face_recognition
    from face_utils import FaceRecognitionSystem
    import cv2
    
    system = FaceRecognitionSystem()
    cap = cv2.VideoCapture(0)
    
    detection_history = {}  # Track faces across frames
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            name, confidence = system.recognize_face(face_encoding, tolerance=0.6)
            
            # Only accept high-confidence matches
            if name != "Unknown" and confidence > 0.75:
                # Track detection
                if name not in detection_history:
                    detection_history[name] = 0
                detection_history[name] += 1
                
                # Draw only high-confidence faces
                top, right, bottom, left = [x*4 for x in face_location]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({confidence:.2f})", (left, top-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('Advanced Filtering', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print detection summary
    print("Detection Summary:")
    for name, count in sorted(detection_history.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {count} detections")
    """)


def example_10_database_integration():
    """Example 10: Integration with database"""
    print("\n" + "="*70)
    print("EXAMPLE 10: Database Integration")
    print("="*70)
    print("""
Example: Store face recognition results in a database

Code:
    import sqlite3
    import face_recognition
    from datetime import datetime
    from face_utils import FaceRecognitionSystem
    
    # Initialize database
    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY,
            person_name TEXT,
            confidence REAL,
            timestamp DATETIME,
            frame_path TEXT
        )
    ''')
    conn.commit()
    
    # Process faces and log
    system = FaceRecognitionSystem()
    
    image = face_recognition.load_image_file("photo.jpg")
    face_encodings = face_recognition.face_encodings(image)
    
    for face_encoding in face_encodings:
        name, confidence = system.recognize_face(face_encoding)
        
        cursor.execute('''
            INSERT INTO detections (person_name, confidence, timestamp)
            VALUES (?, ?, ?)
        ''', (name, confidence, datetime.now()))
    
    conn.commit()
    conn.close()
    
    # Query results
    conn = sqlite3.connect('face_recognition.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT person_name, COUNT(*) as detections FROM detections GROUP BY person_name')
    for row in cursor.fetchall():
        print(f"{row[0]}: {row[1]} detections")
    """)


def main():
    print("\n" + "="*70)
    print(" FACE RECOGNITION SYSTEM - EXAMPLES ".center(70))
    print("="*70)
    
    examples = [
        ("Register people", example_1_register_people),
        ("Recognize a face", example_2_recognize_face),
        ("Real-time webcam", example_3_real_time_webcam),
        ("Process image", example_4_process_image),
        ("Process video", example_5_process_video),
        ("Batch processing", example_6_batch_processing),
        ("Custom tolerance", example_7_custom_tolerance),
        ("Model selection", example_8_model_selection),
        ("Advanced filtering", example_9_advanced_filtering),
        ("Database integration", example_10_database_integration),
        ("Exit", None)
    ]
    
    while True:
        print("\n" + "-"*70)
        print("Select an example to view:")
        for i, (name, _) in enumerate(examples, 1):
            print(f"  {i}. {name}")
        
        try:
            choice = int(input("\nEnter your choice (1-11): "))
            if 1 <= choice <= len(examples):
                if examples[choice-1][1] is None:
                    print("\nGoodbye! 👋\n")
                    break
                examples[choice-1][1]()
                input("\nPress Enter to continue...")
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋\n")
            break


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nError: {str(e)}\n")
