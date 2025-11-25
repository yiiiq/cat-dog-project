import os
from google.cloud import storage
import random
import cv2

# Config
BUCKET_NAME = "image-binary-dataset"
IMG_SIZE = 128
SOURCE_DIR = "data/raw/train"
PROJECT_ID = os.getenv("GCP_PROJECT_ID")

def process_and_upload():
    # Initialize storage client with optional project ID
    storage_client = storage.Client(project=PROJECT_ID) if PROJECT_ID else storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    
    # only upload 1000 cat images like cat.0.jpg and 1000 dog images like dog.0.jpg
    cat_files = [os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR) if f.startswith('cat.')]
    dog_files = [os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR) if f.startswith('dog.')]
    # randomly shuffle the files
    random.shuffle(cat_files)
    random.shuffle(dog_files)
    
    # Check if we have enough images
    num_images = min(1000, len(cat_files), len(dog_files))
    if num_images < 1000:
        print(f"Warning: Only found {len(cat_files)} cat images and {len(dog_files)} dog images. Processing {num_images} pairs.")

    for i in range(num_images): 
        try:
            # 1. Read and Resize
            dog_img = cv2.imread(dog_files[i])
            cat_img = cv2.imread(cat_files[i])
            
            if dog_img is None or cat_img is None:
                print(f"Warning: Failed to read image at index {i}, skipping...")
                continue
                
            dog_img = cv2.resize(dog_img, (IMG_SIZE, IMG_SIZE))
            cat_img = cv2.resize(cat_img, (IMG_SIZE, IMG_SIZE))
            
            # 2. Encode back to jpg in memory
            _, dog_encoded_img = cv2.imencode('.jpg', dog_img)
            _, cat_encoded_img = cv2.imencode('.jpg', cat_img)
            
            # 3. save under dogs/ and cats/ folders in data/processed/
            dog_filename = os.path.basename(dog_files[i])
            cat_filename = os.path.basename(cat_files[i])
            dog_blob = bucket.blob(f"processed/dogs/{dog_filename}")
            cat_blob = bucket.blob(f"processed/cats/{cat_filename}")
            dog_blob.upload_from_string(dog_encoded_img.tobytes(), content_type='image/jpeg')
            cat_blob.upload_from_string(cat_encoded_img.tobytes(), content_type='image/jpeg')
            
            if (i + 1) % 100 == 0:
                print(f"Progress: {i + 1}/{num_images} image pairs uploaded")
        except Exception as e:
            print(f"Error processing image pair at index {i}: {e}")
            continue
    
    print(f"Completed! Uploaded {num_images} cat and dog image pairs.")

if __name__ == "__main__":
    process_and_upload()