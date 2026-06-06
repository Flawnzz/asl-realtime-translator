from pathlib import Path
from collections import deque, Counter

import cv2
import av
import joblib
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from preprocessing import extract_features

# --- CONFIG ---
MODEL_PATH = Path(__file__).parent / "asl_svm_model.joblib"
BOX_FRAC = 0.5              # ROI square side as a fraction of min(frame_w, frame_h)
CALIBRATION_FRAMES = 50     # frames spent memorizing the static background
DEFAULT_MIN_AREA = 1000     # ignore contours smaller than this (noise)
CROP_PADDING = 10           # padding added around the detected hand bbox
DEFAULT_VAR_THRESHOLD = 25  # MOG2 sensitivity
DEFAULT_STABILITY = 8       # frames a letter must persist before it's committed
DEFAULT_CONFIDENCE = 0.50   # frames below this confidence are ignored

# --- PAGE SETUP ---
st.set_page_config(page_title="ASL Real-Time Translator", layout="centered")
st.title("Sign Language Translator 🤟")
st.write(
    "Wait for **READY!**, place your hand in the blue box, and hold each letter "
    "steady until it locks in. Sign SPACE/DEL to edit; lower your hand (NOTHING) "
    "between two of the same letter."
)


# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    svm_model = load_model()
except Exception as e:  # noqa: BLE001 - surface any load failure to the user
    st.error(f"Could not load model at `{MODEL_PATH}`:\n\n{e}")
    st.stop()


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _proba_enabled(model):
    """True if the pipeline's final estimator was trained with probability=True."""
    try:
        return bool(getattr(model[-1], "probability", False))
    except Exception:  # noqa: BLE001
        return False


# Calibrated predict_proba needs SVC(probability=True). Until the model is
# retrained that way, fall back to softmaxed decision-function margins so the
# same 0..1 confidence threshold still applies.
USE_PROBA = _proba_enabled(svm_model)


# --- SIDEBAR TUNING ---
st.sidebar.header("Tuning")
var_threshold = st.sidebar.slider(
    "Background sensitivity", 5, 100, DEFAULT_VAR_THRESHOLD,
    help="MOG2 varThreshold. Lower = more sensitive to motion.",
)
min_area = st.sidebar.slider(
    "Min hand area", 200, 5000, DEFAULT_MIN_AREA, step=100,
    help="Contours smaller than this are ignored as noise.",
)
stability = st.sidebar.slider(
    "Stability (frames to commit)", 1, 20, DEFAULT_STABILITY,
    help="How many consecutive frames a letter must hold before it's typed.",
)
confidence = st.sidebar.slider(
    "Min confidence", 0.0, 1.0, DEFAULT_CONFIDENCE, step=0.05,
    help="Frames below this confidence are ignored instead of voting.",
)
st.sidebar.caption(
    "Confidence source: "
    + ("predict_proba ✅" if USE_PROBA
       else "decision margin (retrain with probability=True for calibrated scores)")
)


class ASLProcessor:
    """Per-frame ASL alphabet recognition with temporal smoothing.

    The heavy work in `recv` is split into small stages (segment -> crop ->
    features -> predict -> smooth -> render) so each can be reasoned about and
    changed independently.
    """

    def __init__(self):
        self.var_threshold = DEFAULT_VAR_THRESHOLD
        self.min_area = DEFAULT_MIN_AREA
        self.stability = DEFAULT_STABILITY
        self.min_confidence = DEFAULT_CONFIDENCE
        self.reset()

    # ---- state ----
    def reset(self):
        """(Re)start background calibration and clear the typed text."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=CALIBRATION_FRAMES,
            varThreshold=self.var_threshold,
            detectShadows=False,
        )
        self.frame_count = 0
        self.recent = deque(maxlen=max(1, self.stability))
        self.last_committed = None
        self.text = ""

    # ---- geometry ----
    @staticmethod
    def roi_box(h, w):
        """Centered square ROI, sized relative to the frame (resolution-agnostic)."""
        side = int(min(h, w) * BOX_FRAC)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        return x0, y0, side

    # ---- pipeline stages ----
    def segment_hand(self, roi_bgr, learning_rate):
        """Return the hand's bounding box within the ROI, or None."""
        fg = self.bg_subtractor.apply(roi_bgr, learningRate=learning_rate)
        _, mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_area:
            return None
        return cv2.boundingRect(largest)

    def predict(self, crop):
        """Return (label, confidence in 0..1) for a grayscale hand crop."""
        feats = extract_features(crop).reshape(1, -1)
        if USE_PROBA:
            proba = svm_model.predict_proba(feats)[0]
        else:
            # No calibrated probabilities: softmax the one-vs-rest decision
            # margins so the same 0..1 confidence threshold still applies.
            proba = _softmax(svm_model.decision_function(feats)[0])
        idx = int(proba.argmax())
        return svm_model.classes_[idx], float(proba[idx])

    def smooth(self, label):
        """Return a label only once a NEW label has been stable for N frames."""
        self.recent.append(label)
        if len(self.recent) < self.recent.maxlen:
            return None
        most_common, count = Counter(self.recent).most_common(1)[0]
        if count == self.recent.maxlen and most_common != self.last_committed:
            self.last_committed = most_common
            return most_common
        return None

    def apply_label(self, label):
        """Turn a committed label into a text-editing action."""
        if label == "SPACE":
            self.text += " "
        elif label == "DEL":
            self.text = self.text[:-1]
        elif label == "NOTHING":
            pass  # acts only as a separator so repeated letters can re-commit
        else:
            self.text += label

    # ---- main entry point ----
    def recv(self, frame):
        try:
            return self._process(frame)
        except Exception:  # noqa: BLE001 - never let a frame error kill the stream
            return frame

    def _process(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        x0, y0, side = self.roi_box(h, w)
        cv2.rectangle(img, (x0, y0), (x0 + side, y0 + side), (255, 0, 0), 2)

        roi_bgr = img[y0:y0 + side, x0:x0 + side]
        gray_roi = gray[y0:y0 + side, x0:x0 + side]

        self.frame_count += 1
        calibrating = self.frame_count <= CALIBRATION_FRAMES

        if calibrating:
            # Keep learning the background; don't try to recognize yet.
            self.bg_subtractor.apply(roi_bgr, learningRate=-1)
            self._banner(img, x0, y0, "MEMORIZING ROOM...", (0, 0, 255))
            self._draw_text(img, h, w)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        self._banner(img, x0, y0, "READY!", (0, 255, 0))
        bbox = self.segment_hand(roi_bgr, learning_rate=0)  # 0 = background locked

        if bbox is not None:
            crop, (hx, hy, hw, hh) = self._crop_hand(gray_roi, bbox, side)
            if crop is not None:
                label, conf = self.predict(crop)
                trusted = conf >= self.min_confidence
                if trusted:
                    committed = self.smooth(label)
                    if committed is not None:
                        self.apply_label(committed)
                # green when trusted, orange when below the confidence gate
                color = (0, 255, 0) if trusted else (0, 165, 255)
                cv2.putText(img, f"{label} {conf:.0%}", (x0, y0 + side + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                cv2.rectangle(img, (x0 + hx, y0 + hy),
                              (x0 + hx + hw, y0 + hy + hh), color, 2)

        self._draw_text(img, h, w)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _crop_hand(self, gray_roi, bbox, side):
        """Apply padding, clamp to the ROI, return (crop, padded_bbox)."""
        hx, hy, hw, hh = bbox
        hx = max(0, hx - CROP_PADDING)
        hy = max(0, hy - CROP_PADDING)
        hw = min(side - hx, hw + CROP_PADDING * 2)
        hh = min(side - hy, hh + CROP_PADDING * 2)
        crop = gray_roi[hy:hy + hh, hx:hx + hw]
        if crop.size == 0:
            return None, (hx, hy, hw, hh)
        return crop, (hx, hy, hw, hh)

    # ---- rendering ----
    @staticmethod
    def _banner(img, x0, y0, text, color):
        cv2.putText(img, text, (x0, max(20, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _draw_text(self, img, h, w):
        shown = self.text[-40:] or "..."
        cv2.rectangle(img, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.putText(img, shown, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


# --- START THE WEBRTC STREAM ---
ctx = webrtc_streamer(
    key="asl-translator",
    video_processor_factory=ASLProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

# Push sidebar values into the running processor (runs on the main thread).
if ctx.video_processor:
    vp = ctx.video_processor
    vp.min_area = min_area
    vp.min_confidence = confidence
    if vp.recent.maxlen != max(1, stability):
        vp.stability = stability
        vp.recent = deque(vp.recent, maxlen=max(1, stability))
    if vp.var_threshold != var_threshold:
        vp.var_threshold = var_threshold
        vp.bg_subtractor.setVarThreshold(var_threshold)

col1, col2 = st.columns(2)
if col1.button("🔄 Recalibrate background"):
    if ctx.video_processor:
        ctx.video_processor.reset()
if col2.button("🧹 Clear text"):
    if ctx.video_processor:
        ctx.video_processor.text = ""
