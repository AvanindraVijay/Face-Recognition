import cv2
import numpy as np
import os
from deepface import DeepFace
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

def capture_video_frames(video_source=0, duration=5):
    """Capture frames from a video for a given duration."""
    cap = cv2.VideoCapture(video_source)
    frames = []
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = frame_rate * duration
    
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def extract_faces_from_frames(frames):
    """Extract the first detected face from each frame."""
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(detected_faces) > 0:
            x, y, w, h = detected_faces[0]
            face = frame[y:y+h, x:x+w]
            faces.append(cv2.resize(face, (160, 160)))
    return faces

def extract_features(image):
    """Extract face embedding from an image using DeepFace."""
    try:
        embedding = DeepFace.represent(img_path=image, model_name='Facenet')[0]['embedding']
        return np.array(embedding)
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

def compare_faces(face_embedding, stored_embeddings, threshold=0.4):
    """Compare the captured face with stored embeddings."""
    for stored_face in stored_embeddings:
        distance = cosine(face_embedding, stored_face)
        if distance < threshold:
            return True  # Match found
    return False

def get_match_confidence(face_embedding, stored_face):
    """Calculate similarity confidence."""
    similarity = cosine_similarity([face_embedding], [stored_face])[0][0]
    return round(similarity * 100, 2)

def load_existing_faces(data_dir):
    """Load and extract features from stored images."""
    stored_embeddings = []
    for filename in os.listdir(data_dir):
        img_path = os.path.join(data_dir, filename)
        embedding = extract_features(img_path)
        if embedding is not None:
            stored_embeddings.append(embedding)
    return stored_embeddings

def main():
    print("Capturing video...")
    frames = capture_video_frames()
    faces = extract_faces_from_frames(frames)
    stored_embeddings = load_existing_faces("data/old_images")
    
    for face in faces:
        face_embedding = extract_features(face)
        if face_embedding is not None:
            if compare_faces(face_embedding, stored_embeddings):
                print("Match found! Proceeding...")
                return
            else:
                new_face_path = f"data/old_images/new_face_{len(stored_embeddings)}.jpg"
                cv2.imwrite(new_face_path, face)
                print(f"New face saved: {new_face_path}")
                stored_embeddings.append(face_embedding)

if __name__ == "__main__":
    main()