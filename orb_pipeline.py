import os
import cv2
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# --- CONFIGURATION ---
# Change these paths to your local Kaggle folders!
DATA_DIR = "Dataset\\asl_alphabet"
VOCAB_SIZE = 1500  # The number of "words" in our visual dictionary
MAX_IMAGES_PER_CLASS = 500  # Keep this low for your first test run!

def build_bovw_pipeline():
    print("Initializing ORB...")
    # nfeatures limits how many points it finds per image to keep math fast
    orb = cv2.ORB_create(nfeatures=300) 
    
    classes = os.listdir(DATA_DIR)
    
    # We need lists to store our data
    all_descriptors = [] # The giant pool for clustering
    image_descriptors = [] # Storing descriptors per image for later
    labels = []
    
    # ==========================================
    # STEP 1: EXTRACT ORB DESCRIPTORS
    # ==========================================
    print("\nStep 1: Extracting ORB features from images...")
    
    for class_name in tqdm(classes, desc="Processing Classes"):
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_dir): continue
            
        img_names = os.listdir(class_dir)[:MAX_IMAGES_PER_CLASS] # LIMITING DATA
        
        for img_name in img_names:
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None: continue
            
            # Find the keypoints and descriptors
            keypoints, descriptors = orb.detectAndCompute(img, None)
            
            if descriptors is not None:
                all_descriptors.extend(descriptors)
                image_descriptors.append(descriptors)
                labels.append(class_name)
            else:
                # If ORB found absolutely nothing (pure black image), skip it
                pass

    # Convert the giant pool to a numpy array (must be float32 for K-Means)
    all_descriptors = np.array(all_descriptors, dtype=np.float32)

    # ==========================================
    # STEP 2: BUILD THE VISUAL DICTIONARY
    # ==========================================
    print(f"\nStep 2: Clustering {len(all_descriptors)} descriptors into {VOCAB_SIZE} visual words...")
    print("This might take a moment...")
    
    # We use MiniBatchKMeans because standard KMeans will crash your RAM on large datasets
    kmeans = MiniBatchKMeans(n_clusters=VOCAB_SIZE, batch_size=3000, random_state=42)
    kmeans.fit(all_descriptors)

    # ==========================================
    # STEP 3: CREATE HISTOGRAMS
    # ==========================================
    print("\nStep 3: Translating images into histograms...")
    
    X_histograms = []
    
    for descs in image_descriptors:
        # Convert descriptors to float32 for prediction
        descs = np.float32(descs)
        
        # Ask K-Means which "pile" each descriptor belongs to
        words = kmeans.predict(descs)
        
        # Count the frequencies of each word to make the histogram
        hist, _ = np.histogram(words, bins=np.arange(VOCAB_SIZE + 1), density=False)
        X_histograms.append(hist)

    X_histograms = np.array(X_histograms)
    y_labels = np.array(labels)

    # ==========================================
    # STEP 4: TRAIN THE CLASSIFIER
    # ==========================================
    print("\nStep 4: Training a *Restricted* Random Forest...")

    # max_depth=15 prevents the trees from memorizing the training data perfectly
    # min_samples_leaf=5 forces it to look for broader, more general patterns
    clf = RandomForestClassifier(
        n_estimators=150, 
        max_depth=15, 
        min_samples_leaf=5, 
        random_state=42, 
        n_jobs=-1
    )
    clf.fit(X_histograms, y_labels)

    print("\n--- Training Complete! ---")

    # ==========================================
    # STEP 5: SAVE EVERYTHING
    # ==========================================
    print("\nSaving vocabulary and model...")
    joblib.dump(kmeans, "orb_vocabulary.joblib")
    joblib.dump(clf, "orb_random_forest.joblib")
    print("Done! You are ready to test the webcam.")

if __name__ == "__main__":
    build_bovw_pipeline()