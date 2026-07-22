#!/usr/bin/env python3
"""
Quick Start Guide for Face Recognition System
"""
import os
import subprocess
import sys


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PROJECT_DIR)
DEFAULT_CONDA_PYTHON = os.path.expanduser(r"C:\Users\prady\miniforge3\envs\face-recog\python.exe")


def get_python_executable():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = os.path.join(conda_prefix, "python.exe") if os.name == "nt" else os.path.join(conda_prefix, "bin", "python")
        if os.path.exists(candidate):
            return candidate

    if os.path.exists(DEFAULT_CONDA_PYTHON):
        return DEFAULT_CONDA_PYTHON

    return sys.executable


def check_dependencies():
    python_exe = get_python_executable()
    try:
        result = subprocess.run(
            [python_exe, "-c", "import cv2, face_recognition; print('ok')"],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
            timeout=60,
        )
        return result.returncode == 0
    except Exception:
        return False


def print_header():
    print("\n" + "="*70)
    print(" FACE RECOGNITION SYSTEM - QUICK START GUIDE ".center(70))
    print("="*70 + "\n")


def print_menu():
    print("\nSelect an option:")
    print("  1. Install dependencies (pip install -r requirements.txt)")
    print("  2. Register a new person")
    print("  3. List registered people")
    print("  4. Delete a registered person")
    print("  5. Run real-time face recognition (webcam)")
    print("  6. Process image file")
    print("  7. Process video file")
    print("  8. Show full documentation")
    print("  9. Exit\n")


def install_dependencies():
    print("\nInstalling Python dependencies...")
    print("This may take a few minutes...\n")

    python_exe = get_python_executable()
    requirements_path = os.path.join(ROOT_DIR, "requirements.txt")

    if not os.path.exists(requirements_path):
        print(f"Error: requirements file not found: {requirements_path}")
        return False

    result = subprocess.run(
        [python_exe, "-m", "pip", "install", "-r", requirements_path],
        cwd=ROOT_DIR,
    )

    if result.returncode != 0:
        print("\n✗ Dependency installation failed.")
        return False

    print("\n✓ Dependencies installed successfully!")
    return True


def register_person():
    print("\n" + "-"*70)
    name = input("Enter person's name: ").strip()

    if not name:
        print("Error: Name cannot be empty")
        return

    samples = input("Number of face samples to capture (default 5): ").strip()
    try:
        samples = int(samples) if samples else 5
    except ValueError:
        samples = 5

    python_exe = get_python_executable()
    script_path = os.path.join(PROJECT_DIR, "register_face.py")

    print(f"\nRegistering {name} ({samples} samples)...")
    print(f"Command: {python_exe} {script_path} --register \"{name}\" --samples {samples}\n")

    subprocess.run([python_exe, script_path, "--register", name, "--samples", str(samples)], cwd=PROJECT_DIR)


def list_people():
    print("\n" + "-"*70)
    print("Registered people:\n")

    python_exe = get_python_executable()
    script_path = os.path.join(PROJECT_DIR, "register_face.py")
    subprocess.run([python_exe, script_path, "--list"], cwd=PROJECT_DIR)


def delete_person():
    print("\n" + "-"*70)
    name = input("Enter person's name to delete: ").strip()

    if not name:
        print("Error: Name cannot be empty")
        return

    confirm = input(f"Delete {name}? (y/n): ").strip().lower()
    if confirm == 'y':
        python_exe = get_python_executable()
        script_path = os.path.join(PROJECT_DIR, "register_face.py")
        subprocess.run([python_exe, script_path, "--delete", name], cwd=PROJECT_DIR)
    else:
        print("Cancelled.")


def run_webcam():
    print("\n" + "-"*70)
    print("Starting real-time face recognition from webcam...")
    print("Press 'q' to quit, 'p' to print detected faces, 's' to save frame\n")

    model = input("Select model (hog=faster, cnn=accurate) [default: hog]: ").strip().lower()
    if model not in ['hog', 'cnn']:
        model = 'hog'

    tolerance = input("Matching tolerance (0.0-1.0, default 0.6): ").strip()
    try:
        tolerance = float(tolerance) if tolerance else 0.6
    except ValueError:
        tolerance = 0.6

    python_exe = get_python_executable()
    script_path = os.path.join(PROJECT_DIR, "face_recognition_real_time.py")

    print(f"\nCommand: {python_exe} {script_path} --webcam --model {model} --tolerance {tolerance}\n")

    subprocess.run([python_exe, script_path, "--webcam", "--model", model, "--tolerance", str(tolerance)], cwd=PROJECT_DIR)


def process_image():
    print("\n" + "-"*70)
    image_path = input("Enter image file path: ").strip()

    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return

    output_path = input("Enter output file path (optional): ").strip()

    python_exe = get_python_executable()
    script_path = os.path.join(PROJECT_DIR, "face_recognition_real_time.py")
    cmd = [python_exe, script_path, "--image", image_path]
    if output_path:
        cmd.extend(["--output", output_path])

    print(f"\nCommand: {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=PROJECT_DIR)


def process_video():
    print("\n" + "-"*70)
    video_path = input("Enter video file path: ").strip()

    if not os.path.exists(video_path):
        print(f"Error: File not found: {video_path}")
        return

    output_path = input("Enter output file path (optional): ").strip()

    python_exe = get_python_executable()
    script_path = os.path.join(PROJECT_DIR, "face_recognition_real_time.py")
    cmd = [python_exe, script_path, "--video", video_path]
    if output_path:
        cmd.extend(["--output", output_path])

    print(f"\nCommand: {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=PROJECT_DIR)


def show_documentation():
    readme_path = os.path.join(PROJECT_DIR, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fh:
            print(fh.read())
    else:
        print("\nREADME.md not found")


def main():
    print_header()

    deps_installed = check_dependencies()
    if not deps_installed:
        print("⚠ Warning: Dependencies not installed. Install them first.\n")

    while True:
        print_menu()
        choice = input("Enter your choice (1-9): ").strip()
        
        if choice == '1':
            deps_installed = install_dependencies()
        elif choice == '2':
            if not deps_installed:
                print("Error: Please install dependencies first (option 1)")
            else:
                register_person()
        elif choice == '3':
            list_people()
        elif choice == '4':
            delete_person()
        elif choice == '5':
            if not deps_installed:
                print("Error: Please install dependencies first (option 1)")
            else:
                run_webcam()
        elif choice == '6':
            if not deps_installed:
                print("Error: Please install dependencies first (option 1)")
            else:
                process_image()
        elif choice == '7':
            if not deps_installed:
                print("Error: Please install dependencies first (option 1)")
            else:
                process_video()
        elif choice == '8':
            show_documentation()
        elif choice == '9':
            print("\nGoodbye! 👋\n")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye! 👋\n")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {str(e)}\n")
        sys.exit(1)
