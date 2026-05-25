import cv2
import av
import joblib
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from skimage.feature import hog

# --- PAGE SETUP ---
st.set_page_config(page_title="ASL Real-Time Translator", layout="centered")
st.title("Sign Language Translator 🤟")
st.write("Wait for the 'READY!' text, then place your hand inside the blue box.")

# --- LOAD THE SVM MODEL ---
@st.cache_resource
def load_model():
    # Loads the final brain we trained on the High-Def HOG features
    return joblib.load('asl_svm_model.joblib')

svm_model = load_model()

# --- THE WEBCAM PROCESSOR ---
class FinalASLProcessor:
    def __init__(self):
        # MOG2 Background Subtractor to memorize the room
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25, detectShadows=False)
        self.frame_count = 0

    def recv(self, frame):
        # Convert the incoming WebRTC frame to an OpenCV array
        img = frame.to_ndarray(format="bgr24")
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Draw the static blue box (300x300)
        cv2.rectangle(img, (100, 100), (400, 400), (255, 0, 0), 2)
        
        # 2. Extract the region inside the blue box
        roi = img[100:400, 100:400]
        gray_roi_full = gray_frame[100:400, 100:400]
        
        # 3. Handle the Learning Rate
        self.frame_count += 1
        # Memorize for 50 frames, then lock it permanently to prevent ghosting
        learning_rate = -1 if self.frame_count <= 50 else 0
        
        fg_mask = self.bg_subtractor.apply(roi, learningRate=learning_rate)
        
        if self.frame_count <= 50:
            cv2.putText(img, "MEMORIZING ROOM...", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        else:
            cv2.putText(img, "READY!", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 4. Clean the mask and find the hand's outer boundary
        _, clean_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Only process if the object is large enough to be a hand
            if cv2.contourArea(largest_contour) > 1000:
                # 5. Get Bounding Box Coordinates
                hx, hy, hw, hh = cv2.boundingRect(largest_contour)
                
                # Add slight padding for a natural Kaggle-style crop
                padding = 10
                hx = max(0, hx - padding)
                hy = max(0, hy - padding)
                hw = min(300 - hx, hw + padding * 2)
                hh = min(300 - hy, hh + padding * 2)
                
                # 6. The Tight Crop (Extracting raw pixels, not the mask)
                hand_crop = gray_roi_full[hy:hy+hh, hx:hx+hw]
                
                if hand_crop.size > 0:
                    # 7. High-Def Resize matching the training script
                    resized_hand = cv2.resize(hand_crop, (128, 128))
                    
                    # 8. Extract HOG Features
                    features = hog(resized_hand, 
                                   orientations=9, 
                                   pixels_per_cell=(16, 16),
                                   cells_per_block=(2, 2), 
                                   block_norm='L2-Hys', 
                                   visualize=False)
                    
                    # 9. Predict using the SVM
                    # reshape(1, -1) turns the 1D list into the 2D array Scikit-Learn requires
                    prediction = svm_model.predict(features.reshape(1, -1))[0]
                    
                    # Display the prediction heavily on the screen
                    cv2.putText(img, f"Prediction: {prediction}", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    
                    # Draw a green tracking box around the hand inside the blue box
                    cv2.rectangle(img, (100 + hx, 100 + hy), (100 + hx + hw, 100 + hy + hh), (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- START THE WEBRTC STREAM ---
webrtc_streamer(
    key="asl-translator", 
    video_processor_factory=FinalASLProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)