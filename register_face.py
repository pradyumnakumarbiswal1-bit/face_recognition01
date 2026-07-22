import cv2
import os
import argparse
from face_utils import FaceRecognitionSystem


def register_person(person_name, num_samples=5, known_faces_dir='known_faces', tolerance=0.5):
    system = FaceRecognitionSystem(known_faces_dir=known_faces_dir)
    person_dir = os.path.join(known_faces_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    captured = 0
    skipped = 0
    
    print(f"\n{'='*50}")
    print(f"Registering: {person_name}")
    print(f"{'='*50}")
    print(f"Recognition tolerance: {tolerance}")
    print(f"Instructions:")
    print(f"  - Press 'c' to capture a face image")
    print(f"  - Press 'q' to quit")
    print(f"  - Position your face in the blue rectangle")
    print(f"  - Capture {num_samples} images with different angles")
    print(f"{'='*50}\n")
    
    while captured < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from webcam")
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f'Captured: {captured}/{num_samples}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Skipped: {skipped}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if len(faces) == 0:
            cv2.putText(frame, 'No face detected', (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, 'Press C to capture', (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(f'Registering: {person_name}', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if len(faces) > 0:
                x, y, w, h = faces[0]
                filename = f'{person_name}_{captured}.jpg'
                filepath = os.path.join(person_dir, filename)
                cv2.imwrite(filepath, frame)
                print(f"✓ Captured: {filename}")
                captured += 1
            else:
                print("✗ No face detected, skipping...")
                skipped += 1
        elif key == ord('q'):
            print("Registration cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{'='*50}")
    print(f"✓ Captured {captured} images for {person_name}")
    print(f"{'='*50}\n")
    print("Generating face encodings...")
    system.generate_encodings_from_folder()
    print(f"\n✓ {person_name} registered successfully!\n")


def list_registered_people(known_faces_dir='known_faces'):
    system = FaceRecognitionSystem(known_faces_dir=known_faces_dir)
    people = system.get_registered_people()
    if not people:
        print("No registered people found.")
        return
    print(f"\n{'='*50}")
    print("Registered People:")
    print(f"{'='*50}")
    for person_name, image_count in people:
        print(f"  • {person_name}: {image_count} image(s)")
    print(f"{'='*50}\n")


def delete_person(person_name, known_faces_dir='known_faces'):
    system = FaceRecognitionSystem(known_faces_dir=known_faces_dir)
    person_dir = os.path.join(known_faces_dir, person_name)
    if not os.path.exists(person_dir):
        print(f"Error: {person_name} not found.")
        return
    for image_name in os.listdir(person_dir):
        os.remove(os.path.join(person_dir, image_name))
    os.rmdir(person_dir)
    print(f"✓ Deleted {person_name}")
    print("Regenerating encodings...")
    system.generate_encodings_from_folder()
    print("✓ Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Register new people for face recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python register_face.py --register "John Doe"
  python register_face.py --register "Jane Smith" --samples 10
  python register_face.py --list
  python register_face.py --delete "John Doe"
        '''
    )
    
    parser.add_argument('--register', type=str, help='Register a new person (provide name)')
    parser.add_argument('--samples', type=int, default=5, help='Number of face samples to capture (default: 5)')
    parser.add_argument('--tolerance', type=float, default=0.5, help='Recognition tolerance to display during registration (default: 0.5)')
    parser.add_argument('--list', action='store_true', help='List all registered people')
    parser.add_argument('--delete', type=str, help='Delete a registered person')
    parser.add_argument('--dir', type=str, default='known_faces', help='Directory for known faces')
    
    args = parser.parse_args()
    
    if args.register:
        register_person(args.register, num_samples=args.samples, known_faces_dir=args.dir, tolerance=args.tolerance)
    elif args.list:
        list_registered_people(known_faces_dir=args.dir)
    elif args.delete:
        confirm = input(f"Delete {args.delete}? (y/n): ")
        if confirm.lower() == 'y':
            delete_person(args.delete, known_faces_dir=args.dir)
    else:
        parser.print_help()
