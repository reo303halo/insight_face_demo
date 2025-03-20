import cv2
import numpy as np
from insightface.app import FaceAnalysis
# Base code: https://github.com/deepinsight/insightface/blob/master/examples/face_recognition/insightface_app.py

# Initialize face analysis model
def initialize_model():
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']) # For CPU
    #app = FaceAnalysis(name='buffalo_l', providers=['CoreMLExecutionProvider'])  # For GPU
    app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU
    return app


# Extract face embedding from an image
def get_face_embedding(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    faces = model.get(img)
    
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
    image1_path = "photos/roy_passport.jpeg" #Passport photo
    model = initialize_model()

    total_score = 0
    size = 10
    # Photo 7 is from profile view
    # Photo 8 is with bigger sunglasses

    for i in range(0, size):
        image2_path = f"photos/roy{i+1}.jpeg"

        try:
            print(f"Comparing roy_passport with roy{i+1}")
            # Get embeddings
            emb1 = get_face_embedding(image1_path, model)
            emb2 = get_face_embedding(image2_path, model)
            
            # Compare faces
            similarity_score = compare_faces(emb1, emb2)
            
            similarity_score_percent = similarity_score * 100
            total_score += similarity_score_percent

            print(f"Similarity Score: {similarity_score_percent:.2f}%\n")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    

    average_score = total_score / size
    print(f"Average score of test is {average_score:.3f}% similarity")

    result = check_match(average_score)
    print(result)

if __name__ == "__main__":
    main()
