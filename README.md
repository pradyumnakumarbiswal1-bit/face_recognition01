# Face Detection & Recognition (Task 3) — Synrecxhub AI Internship

Project: Face recognition demo and registration pipeline

This project was developed as part of the AI internship under Synrecxhub (Task 3). It provides a small face-registration and real-time face-recognition system using a webcam, OpenCV, and the `face_recognition` library.

Contents
- Quick description and goals
- Prerequisites
- Files of interest
- How to register faces (webcam)
- How to run real-time recognition (webcam)
- Commands to regenerate encodings, remove cache
- Troubleshooting notes

Quick description
- Capture face samples via webcam into `known_faces/<PersonName>/`.
- Generate and cache 128-D face encodings in `face_encodings.pkl` for fast recognition.
- Run real-time recognition using `face_recognition_real_time.py` (default model: `cnn`, tolerance: `0.5`).

Prerequisites
- Python 3.8+ (project used Python 3.11)
- Virtual environment recommended (project venv at `.venv/` if used)
- Required Python packages: OpenCV, numpy (compatible), face_recognition (dlib + models). Note: `requirements.txt` was removed from the repo; install packages manually in your venv if needed.

Activate venv (PowerShell)
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .venv\Scripts\Activate.ps1
```

Install common packages (example)
```powershell
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install face-recognition==1.3.0
```

Files of interest
- [face_utils.py](face_utils.py) — encoding generation, saving/loading, recognition helpers
- [register_face.py](register_face.py) — register persons via webcam and generate encodings
- [face_recognition_real_time.py](face_recognition_real_time.py) — run webcam recognition
- [known_faces/](known_faces/) — per-person image folders (created by registration)
- [face_encodings.pkl](face_encodings.pkl) — cached encodings (kept for speed)

Register faces (webcam)
- Purpose: capture samples for a named person and update the encodings cache.
- Example: register user "Pradyumna" with 20 samples, tolerance 0.5
```powershell
python register_face.py --register "Pradyumna" --samples 20 --tolerance 0.5
```
- Notes:
  - This creates `known_faces/Pradyumna/` and saves captured images there.
  - After capture, `register_face.py` will call encoding generation to update `face_encodings.pkl`.
  - Use `--samples` to increase sample count for better coverage of angles/lighting.
  - Use `--list` to list registered people, and `--delete <Name>` to remove a person.

Run real-time recognition (webcam)
- Explicit webcam command (recommended):
```powershell
python face_recognition_real_time.py --webcam --model cnn --tolerance 0.5
```
- Additional examples:
```powershell
python face_recognition_real_time.py --webcam
python face_recognition_real_time.py --webcam --model cnn
python face_recognition_real_time.py --webcam --tolerance 0.4
```
- Process a video file:
```powershell
python face_recognition_real_time.py --video input.mp4 --output output.mp4
```
- Process an image:
```powershell
python face_recognition_real_time.py --image test.jpg --output result.jpg
```
- Options:
  - `--webcam`: use live webcam feed
  - `--video <path>`: process video file
  - `--image <path>`: process single image
  - `--model`: `cnn` (better accuracy) or `hog` (faster)
  - `--tolerance`: face matching threshold (lower=stricter, default 0.5)
  - `--output`: save result to file (for image/video modes)

Regenerate encodings manually
- If you add or remove images in `known_faces/` and want to force a regeneration:
```powershell
python -c "from face_utils import FaceRecognitionSystem; FaceRecognitionSystem().generate_encodings_from_folder()"
```

Remove encoding cache (force rebuild on next run)
```powershell
Remove-Item face_encodings.pkl -Force
```

Best practices to improve accuracy
- Capture 20–50 diverse samples per person (different angles, distances, lighting)
- Use `--samples` to increase images at registration
- Keep `--model cnn` for better detection at different angles
- Tune `--tolerance` (0.4–0.6 range common)

Troubleshooting
- If recognition says "unknown" even though images exist: ensure `face_encodings.pkl` exists or regenerate encodings; run the regeneration command above.
- If OpenCV / numpy import errors occur, reinstall matching numpy and OpenCV wheel versions in the venv.

License & Notes
- This repo is a learning/demo project for the Synrecxhub AI internship (Task 3). Adjust and extend for production carefully (security, privacy, and model performance considerations).
