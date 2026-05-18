import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import joblib
from skimage.feature import hog

@st.cache_resource
def load_model():
    return joblib.load("asl_svm_model.joblib")

model = load_model()

def process_frame(frame_roi):
    hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
    
    resized = cv2.resize(clean_mask, (64, 128))
    features = hog(resized, orientations=9, pixels_per_cell=(8, 8), 
                   cells_per_block=(2, 2), visualize=False)
    
    return model.predict([features])[0]

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Define bounding box for hand placement
        x1, y1, x2, y2 = 100, 100, 400, 400
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        roi = img[y1:y2, x1:x2]
        
        try:
            prediction = process_frame(roi)
            cv2.putText(img, f"Prediction: {prediction}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception:
            pass
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("ASL Translator")
st.write("Place your hand inside the blue box.")

webrtc_streamer(
    key="asl-translator", 
    video_processor_factory=VideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)