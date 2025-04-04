import os
from utils.image_preprocessing import preprocess_image
from utils.feature_extraction import extract_features
from utils.comparison_utils import compare_faces, get_match_confidence

# Define paths
data_dir = "data"
old_images_dir = os.path.join(data_dir, "old_images")
current_image_path = os.path.join(data_dir, "current_images", "current_person.jpg")

# Preprocess and extract features for the current image
print("Processing current image...")
current_image = preprocess_image(current_image_path)
current_embedding = extract_features(current_image_path)

if current_embedding is None:
    print("Error: Could not extract features from the current image.")
    exit()

# Iterate through old images and compare
print("Comparing with old images...")
best_match = None
best_confidence = 0

for old_image_name in os.listdir(old_images_dir):
    old_image_path = os.path.join(old_images_dir, old_image_name)
    old_embedding = extract_features(old_image_path)
    
    if old_embedding is not None:
        confidence = get_match_confidence(current_embedding, old_embedding)
        print(f"Match confidence with {old_image_name}: {confidence}%")
        
        if confidence > best_confidence:
            best_confidence = confidence
            best_match = old_image_name

# Display final result
if best_match and best_confidence > 50:  # Threshold can be adjusted
    print(f"Best match found: {best_match} with {best_confidence}% confidence.")
else:
    print("No strong match found in old images.")
