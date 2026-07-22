import cv2
import face_recognition
import numpy as np
import argparse
from face_utils import FaceRecognitionSystem, draw_face_box_with_label


class RealTimeFaceRecognition:
    def __init__(self, known_faces_dir='known_faces', model='cnn'):
        self.system = FaceRecognitionSystem(known_faces_dir=known_faces_dir)
        self.model = model
        self.frame_count = 0
        self.process_every_n_frames = 3
        self.last_face_results = []
    
    def process_frame(self, frame, tolerance=0.5, show_confidence=True):
        self.frame_count += 1
        display_frame = frame.copy()
        if self.frame_count % self.process_every_n_frames != 0:
            for top, right, bottom, left, name, confidence in self.last_face_results:
                draw_face_box_with_label(display_frame, top, right, bottom, left, name, confidence)
            return display_frame
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame, model=self.model)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_locations = [(top*4, right*4, bottom*4, left*4) 
                         for top, right, bottom, left in face_locations]
        self.last_face_results = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            top, right, bottom, left = face_location
            name, confidence = self.system.recognize_face(face_encoding, tolerance=tolerance)
            self.last_face_results.append((top, right, bottom, left, name, confidence))
            draw_face_box_with_label(display_frame, top, right, bottom, left, name,
                                     confidence if show_confidence else None)
        return display_frame
    
    def run_webcam(self, tolerance=0.5, show_confidence=True):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"\n{'='*60}")
        print("Real-Time Face Recognition")
        print(f"{'='*60}")
        print(f"Resolution: {frame_width}x{frame_height}")
        print(f"Model: {self.model}")
        print(f"Known faces: {len(self.system.known_names)}")
        print(f"\nControls:")
        print(f"  - 'q' to quit")
        print(f"  - 's' to save current frame")
        print(f"  - 'p' to print detected faces")
        print(f"{'='*60}\n")
        frame_num = 0
        detected_faces_count = {}
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read from webcam")
                break
            frame = cv2.flip(frame, 1)
            processed_frame = self.process_frame(frame, tolerance=tolerance, 
                                               show_confidence=show_confidence)
            frame_num += 1
            cv2.putText(processed_frame, f'FPS: {frame_num}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(processed_frame, f'Press H for help', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow('Face Recognition', processed_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting...")
                break
            elif key == ord('s'):
                filename = f'face_detection_{frame_num}.jpg'
                cv2.imwrite(filename, processed_frame)
                print(f"✓ Saved: {filename}")
            elif key == ord('p'):
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small_frame, model=self.model)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                print(f"\nDetected {len(face_encodings)} face(s):")
                for i, face_encoding in enumerate(face_encodings):
                    name, confidence = self.system.recognize_face(face_encoding, tolerance=tolerance)
                    print(f"  {i+1}. {name} (confidence: {confidence:.2f})")
            elif key == ord('h'):
                print("\n" + "="*60)
                print("Controls:")
                print("  - 'q' to quit")
                print("  - 's' to save current frame")
                print("  - 'p' to print detected faces")
                print("  - 'h' to show this help")
                print("="*60 + "\n")
        cap.release()
        cv2.destroyAllWindows()
    
    def process_image(self, image_path, tolerance=0.5, output_path=None):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image: {image_path}")
            return
        print(f"Processing: {image_path}")
        processed_frame = self.process_frame(frame, tolerance=tolerance)
        cv2.imshow('Face Recognition - Image', processed_frame)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if output_path:
            cv2.imwrite(output_path, processed_frame)
            print(f"✓ Saved: {output_path}")
    
    def process_video(self, video_path, tolerance=0.5, output_path=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video: {video_path}")
            return
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing video: {video_path}")
        print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}, Total frames: {total_frames}")
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            processed_frame = self.process_frame(frame, tolerance=tolerance)
            progress = (frame_count / total_frames) * 100
            cv2.putText(processed_frame, f'Progress: {progress:.1f}%', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.imshow('Processing Video', processed_frame)
            if output_path:
                out.write(processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count}/{total_frames} frames...")
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        print(f"✓ Finished processing {frame_count} frames")
        if output_path:
            print(f"✓ Saved: {output_path}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Real-time face detection and recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python face_recognition_real_time.py --webcam
  python face_recognition_real_time.py --webcam --tolerance 0.5
  python face_recognition_real_time.py --image test.jpg --output result.jpg
  python face_recognition_real_time.py --video input.mp4 --output output.mp4
  python face_recognition_real_time.py --webcam --model cnn
        '''
    )
    
    parser.add_argument('--webcam', action='store_true', help='Use webcam for real-time recognition')
    parser.add_argument('--image', type=str, help='Process a single image file')
    parser.add_argument('--video', type=str, help='Process a video file')
    parser.add_argument('--tolerance', type=float, default=0.5, help='Face matching tolerance (0.0-1.0, lower=stricter)')
    parser.add_argument('--model', choices=['hog', 'cnn'], default='cnn', help='Face detection model')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--dir', type=str, default='known_faces', help='Directory containing known faces')
    parser.add_argument('--no-confidence', action='store_true', help='Hide confidence scores')
    
    args = parser.parse_args()
    
    # Initialize system
    recognizer = RealTimeFaceRecognition(known_faces_dir=args.dir, model=args.model)
    
    if args.webcam:
        recognizer.run_webcam(tolerance=args.tolerance, 
                            show_confidence=not args.no_confidence)
    elif args.image:
        recognizer.process_image(args.image, tolerance=args.tolerance, output_path=args.output)
    elif args.video:
        recognizer.process_video(args.video, tolerance=args.tolerance, output_path=args.output)
    else:
        parser.print_help()
