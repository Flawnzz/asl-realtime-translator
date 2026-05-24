import cv2
import av
import numpy as np
import joblib
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from skimage.feature import hog

# Load your original, highly accurate SVM model
@st.cache_resource
def load_model():
    return joblib.load("asl_svm_model.joblib")

svm_model = load_model()

class FinalASLProcessor:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
        self.frame_count = 0  # <-- 1. Add a frame counter
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1 # <-- 2. Count every frame that passes
        
        x1, y1, x2, y2 = 100, 100, 400, 400
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        roi = img[y1:y2, x1:x2]
        
        # --- 3. THE LEARNING RATE FIX ---
        if self.frame_count < 50:
            # First ~2 seconds: Actively memorize the room (learningRate=0.1)
            mask = self.bg_subtractor.apply(roi, learningRate=0.1)
            cv2.putText(img, "MEMORIZING... KEEP HAND OUT", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Skip predictions while memorizing
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        else:
            # After 2 seconds: FREEZE the background (learningRate=0)
            # It will no longer erase your hand when you hold still!
            mask = self.bg_subtractor.apply(roi, learningRate=0)
            cv2.putText(img, "READY!", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        # 3. Clean up the silhouette (remove static/noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # We need the raw grayscale video for the final crop
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if contours:
            # Assume the largest white blob in the blue box is your hand
            biggest_contour = max(contours, key=cv2.contourArea)
            
            # --- THE TIGHT CROP FIX ---
            # Get the exact X, Y coordinates and width/height of your hand
            hx, hy, hw, hh = cv2.boundingRect(biggest_contour)
            
            # Saftey check: Ignore tiny glitches that aren't big enough to be a hand
            if hw > 50 and hh > 50:
                # --- THE RAW TRACKING FIX ---
                # Add 20 pixels of padding so we don't chop off fingertips
                # (We use max/min to ensure we don't accidentally crop outside the video)
                pad = 20
                y1_crop = max(0, hy - pad)
                y2_crop = min(roi.shape[0], hy + hh + pad)
                x1_crop = max(0, hx - pad)
                x2_crop = min(roi.shape[1], hx + hw + pad)

                # Crop the RAW video using the MOG2 coordinates (No more black background!)
                raw_hand_crop = gray_roi[y1_crop:y2_crop, x1_crop:x2_crop]
                
                # --- IMPORTANT: MATCH YOUR TRAINING SETTINGS HERE ---
                resized_hand = cv2.resize(raw_hand_crop, (64, 128)) 
                
                features = hog(resized_hand, orientations=9, pixels_per_cell=(8, 8), 
                               cells_per_block=(2, 2), visualize=False)
                # ----------------------------------------------------
                
                features_reshaped = features.reshape(1, -1)
                
                try:
                    prediction = svm_model.predict(features_reshaped)[0]
                    cv2.putText(img, f"Sign: {prediction}", (100, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                except Exception as e:
                    pass

                # Picture-in-Picture: Watch how different this looks now!
                cutout_bgr = cv2.cvtColor(resized_hand, cv2.COLOR_GRAY2BGR)
                img[0:128, 0:64] = cutout_bgr 

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("Final ASL Translator (MOG2 + SVM)")
st.write("**INSTRUCTIONS:**")
st.write("1. Keep your hand **OUT** of the blue box when the video starts.")
st.write("2. Wait 3 seconds for the AI to memorize your room.")
st.write("3. Put your hand in the box to translate!")
webrtc_streamer(key="final-asl", video_processor_factory=FinalASLProcessor)