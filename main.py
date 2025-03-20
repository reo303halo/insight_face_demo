import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis
# Base code: https://github.com/deepinsight/insightface/blob/master/examples/face_recognition/insightface_app.py

start_time = time.time()  # Start timer
# Initialize face analysis model globally (so it's not reloaded in case of multiple tests)
#FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']) # For CPU
MODEL = FaceAnalysis(name='buffalo_l', providers=['CoreMLExecutionProvider'])  # Use GPU
MODEL.prepare(ctx_id=0)  # ctx_id=-1 for CPU, 0 for GPU


# Extract face embedding from an image
def get_face_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    faces = MODEL.get(img)
    
    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")
    
    return faces[0].embedding


# Compare two embeddings using cosine similarity
def compare_faces(emb1, emb2):
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity


# Helper function
def check_match(similarity):
    match similarity: # Match case available in Python 3.10+
        case similarity if similarity >= 90:
            return "✅ Strong match! Same person."
        case similarity if 75 <= similarity < 90:
            return "🟡 Possible match. Looks similar, but not certain."
        case similarity if 50 <= similarity < 75:
            return "⚠️ Weak match. Faces share some features, but might be different people."
        case _:
            return "❌ No match. Different persons."
        
        
def main():
    

    image1_path = "photos/roy_passport.jpeg"  # Passport photo
    image_paths = [f"photos/roy{i+1}.jpeg" for i in range(17)]  # Precompute paths

    try:
        emb1 = get_face_embedding(image1_path)  # Extract once
    except Exception as e:
        print(f"Error processing passport image: {e}")
        return

    similarities = []
    
    for image2_path in image_paths:
        try:
            print(f"Comparing {image1_path} with {image2_path}")

            emb2 = get_face_embedding(image2_path)
            similarity_score = compare_faces(emb1, emb2) * 100  # Convert to percentage
            
            similarities.append(similarity_score)
            print(f"Similarity Score: {similarity_score:.2f}%\n")
        
        except Exception as e:
            print(f"Error processing {image2_path}: {e}")

    if similarities:
        avg_score = np.mean(similarities)  # Use NumPy for efficient mean calculation
        print(f"Average similarity: {avg_score:.3f}%")
        print(check_match(avg_score))


    end_time = time.time()  # End timer
    total_time = end_time - start_time

    print(f"Total Execution Time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
