import cv2
import numpy as np
import av
import joblib
import streamlit as st
from streamlit_webrtc import webrtc_streamer

# Load your new Bag of Visual Words models
@st.cache_resource
def load_models():
    kmeans = joblib.load("orb_vocabulary.joblib")
    clf = joblib.load("orb_random_forest.joblib")
    return kmeans, clf

kmeans, clf = load_models()
VOCAB_SIZE = 1500

class VideoProcessor:
    def __init__(self):
        # Initialize ORB once when the webcam starts
        self.orb = cv2.ORB_create(nfeatures=300)
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Define the Blue Box
        x1, y1, x2, y2 = 100, 100, 400, 400
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        roi = img[y1:y2, x1:x2]
        
        # 2. Convert to Grayscale (No more finicky skin masking!)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 3. Extract ORB features from the live frame
        keypoints, descriptors = self.orb.detectAndCompute(gray_roi, None)
        
        if descriptors is not None:
            # Draw the keypoints on your hand so you can see the math working
            cv2.drawKeypoints(roi, keypoints, roi, color=(0, 255, 0), flags=0)
            
            # 4. Translate the points into a Histogram
            descriptors = np.float32(descriptors)
            words = kmeans.predict(descriptors)
            hist, _ = np.histogram(words, bins=np.arange(VOCAB_SIZE + 1), density=False)
            
            # 5. Predict the letter using Random Forest
            prediction = clf.predict([hist])[0]
            
            # 6. Display the prediction on the screen
            cv2.putText(img, f"Sign: {prediction}", (100, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("ASL Translator (ORB Keypoints)")
st.write("Place your hand in the blue box. Watch the green keypoints track your hand!")
webrtc_streamer(key="asl-stream", video_processor_factory=VideoProcessor)