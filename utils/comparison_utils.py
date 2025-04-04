from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compare_faces(face1_embedding, face2_embedding, threshold=0.4):
    """Compare two face embeddings using cosine distance."""
    distance = cosine(face1_embedding, face2_embedding)
    return distance < threshold  # Returns True if faces match


def get_match_confidence(face1_embedding, face2_embedding):
    """Calculate the match confidence as a percentage."""
    similarity = cosine_similarity([face1_embedding], [face2_embedding])[0][0]
    confidence = similarity * 100
    return round(confidence, 2)


if __name__ == "__main__":
    # Test embeddings (random example)
    face1 = np.random.rand(128)
    face2 = np.random.rand(128)
    
    match = compare_faces(face1, face2)
    confidence = get_match_confidence(face1, face2)
    
    print(f"Faces Match: {match}")
    print(f"Match Confidence: {confidence}%")