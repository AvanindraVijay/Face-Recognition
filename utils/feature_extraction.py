import cv2
import dlib
import numpy as np
from deepface import DeepFace

# Load dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def load_image(image_path):
    """Load an image from a given path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    return image

def align_face(image, rect):
    """Align face using dlib's facial landmarks."""
    shape = predictor(image, rect)
    
    # Get left and right eye coordinates
    left_eye = (shape.part(36).x, shape.part(36).y)
    right_eye = (shape.part(45).x, shape.part(45).y)
    
    # Compute angle between eyes
    dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Get center of eyes
    center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    # Rotate image to align eyes horizontally
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    aligned_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))

    return aligned_image

def extract_features(image):
    """Extract face embeddings from an image using DeepFace."""
    try:
        embedding = DeepFace.represent(img_path=image, model_name='Facenet')[0]['embedding']
        return np.array(embedding)
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

def extract_faces_from_video(video_source=0, duration=5):
    """Capture video for a specified duration and extract aligned faces."""
    cap = cv2.VideoCapture(video_source)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = frame_rate * duration
    faces = []

    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = detector(gray)

        for rect in detected_faces:
            aligned_face = align_face(frame, rect)
            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            face = aligned_face[y:y+h, x:x+w]
            faces.append(cv2.resize(face, (160, 160)))

    cap.release()
    return faces

if __name__ == "__main__":
    print("Capturing video and extracting features...")
    faces = extract_faces_from_video()
    embeddings = [extract_features(face) for face in faces if face is not None]
    print(f"Extracted {len(embeddings)} feature embeddings.")
