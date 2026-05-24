import os
import cv2
import numpy as np
from skimage.feature import hog
import joblib
from tqdm import tqdm

def preprocess_and_extract(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    # Skin Color Thresholding in HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Morphological cleaning
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
    
    resized = cv2.resize(clean_mask, (64, 128)) 
    
    # Extract HOG features
    features = hog(resized, orientations=9, pixels_per_cell=(8, 8), 
                   cells_per_block=(2, 2), visualize=False)
    return features

def process_directory(dataset_path):
    X, y = [], []
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    for class_name in tqdm(classes, desc=f"Processing {os.path.basename(dataset_path)}"):
        class_dir = os.path.join(dataset_path, class_name)
        label = class_name.upper()
        
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                features = preprocess_and_extract(os.path.join(class_dir, img_name))
                if features is not None:
                    X.append(features)
                    y.append(label)
    return X, y

if __name__ == "__main__":
    # Ensure these paths match where you put your downloaded data
    path1 = "Dataset\\asl_dataset" 
    path2 = "Dataset\\asl_alphabet"
    
    print("Extracting Dataset 1...")
    X1, y1 = process_directory(path1)
    
    print("Extracting Dataset 2...")
    X2, y2 = process_directory(path2)
    
    X_all = np.array(X1 + X2)
    y_all = np.array(y1 + y2)
    
    print("Saving extracted features...")
    joblib.dump({'X': X_all, 'y': y_all}, "extracted_features.joblib")
    print("Done! Proceed to Step 2.")