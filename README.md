
# Face Recognition System

This project implements a face recognition system that can extract, compare, and match faces from images and video frames using OpenCV, DeepFace, and Dlib.

## Features

- **Face Detection & Alignment**: Uses Dlib and OpenCV to detect and align faces.
- **Feature Extraction**: Extracts face embeddings using DeepFace's `Facenet` model.
- **Video Processing**: Captures video frames and detects faces in real-time.
- **Face Comparison**: Matches a captured face against stored embeddings using cosine similarity.
- **Face Storage**: Stores new faces for future comparisons.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/AvanindraVijay/Face-Recognition.git
   cd Face-Recognition
Install dependencies:

sh
Copy
Edit
pip install -r requirements.txt
Download shape_predictor_68_face_landmarks.dat (required for Dlib):

Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Extract and place it in the models/ directory.

Usage
Run Face Extraction from Video
sh
Copy
Edit
python feature_extraction.py
Run Image Preprocessing
sh
Copy
Edit
python image_preprocessing.py
Compare Faces from Video
sh
Copy
Edit
python comparison_utils.py
Directory Structure
bash
Copy
Edit
Face-Recognition/
│── models/                     # Pre-trained model files
│── data/                        # Stored face images
│── face_rec_env/                # Virtual environment (not included in Git)
│── feature_extraction.py        # Extracts features from faces
│── image_preprocessing.py       # Preprocessing functions
│── comparison_utils.py          # Face comparison logic
│── requirements.txt             # Python dependencies
│── README.md                    # Project documentation
