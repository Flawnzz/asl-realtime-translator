# ASL Real-Time Translator — Pipeline

Two pipelines — offline **training** and live **inference** — share one
preprocessing function (`preprocessing.extract_features`) so they can never
drift apart. GitHub renders the Mermaid diagram below automatically.

```mermaid
flowchart TD
    %% ---------------- TRAINING ----------------
    subgraph TRAIN["TRAINING (offline)"]
        direction TB
        D1["Kaggle ASL dataset<br/>(A–Z, SPACE, DEL, NOTHING)"]
        D2["Sample ~800–1000<br/>images / class (balanced)"]
        D3["BGR → grayscale"]
        D4["Segment hand →<br/>tight bbox crop"]
        T1["StandardScaler → PCA(0.95)<br/>→ SVC(rbf)"]
        T2["asl_svm_model.joblib"]
        D1 --> D2 --> D3 --> D4
        T1 --> T2
    end

    %% ---------------- SHARED ----------------
    subgraph SHARED["SHARED — preprocessing.py"]
        P0["extract_features(gray_crop)<br/>square_pad → 128×128 resize → HOG (1764-D)"]
    end

    %% ---------------- INFERENCE ----------------
    subgraph INFER["LIVE INFERENCE (per frame)"]
        direction TB
        F1["WebRTC frame → BGR ndarray"]
        F2["Center ROI box<br/>(BOX_FRAC of frame)"]
        F3{"calibrating?<br/>(frame ≤ 50)"}
        F4["MOG2 learn bg<br/>'MEMORIZING ROOM...'"]
        F5["MOG2 apply (locked)<br/>→ threshold → contours"]
        F6{"contour ≥ min_area?"}
        F7["pad + clamp bbox → gray crop"]
        F9{"USE_PROBA?"}
        F9a["predict_proba"]
        F9b["softmax(decision_function)"]
        F10["(label, confidence)"]
        F11{"confidence ≥ min_confidence?"}
        F12["smooth(): deque,<br/>stable N frames & new?"]
        F13["apply_label():<br/>letter / SPACE / DEL / NOTHING"]
        F14["render overlay + output<br/>(box, label %, typed text)"]

        F1 --> F2 --> F3
        F3 -- yes --> F4 --> F14
        F3 -- no --> F5 --> F6
        F6 -- no --> F14
        F6 -- yes --> F7
        F9 -- proba --> F9a --> F10
        F9 -- margin --> F9b --> F10
        F10 --> F11
        F11 -- no --> F14
        F11 -- yes --> F12
        F12 -- committed --> F13 --> F14
        F12 -- not stable --> F14
    end

    %% ---- cross-pipeline links through the shared module ----
    D4 -- train --> P0 --> T1
    F7 -- crop --> P0 --> F9
    T2 -. loaded by predict() .-> F9

    classDef train fill:#1e3a8a,stroke:#cbd5e1,color:#fff;
    classDef shared fill:#14532d,stroke:#cbd5e1,color:#fff;
    classDef infer fill:#7c2d12,stroke:#cbd5e1,color:#fff;
    classDef dec fill:#f59e0b,stroke:#222,color:#000;
    class D1,D2,D3,D4,T1,T2 train;
    class P0 shared;
    class F1,F2,F4,F5,F7,F9a,F9b,F10,F12,F13,F14 infer;
    class F3,F6,F9,F11 dec;
```

## Key points
- **Shared green node** (`preprocessing.extract_features`) is used by *both*
  training and inference — this is what guarantees the live square-pad/HOG
  matches what the model learned.
- **Confidence gate** (`F11`) drops uncertain frames before they vote.
- **Temporal smoothing** (`F12`) only commits a letter after it's stable for N frames.
- **`apply_label`** (`F13`) turns SPACE/DEL/NOTHING into real text edits, making it
  a translator rather than a per-frame classifier.
- The model currently runs the **softmax(decision_function)** branch; it switches to
  calibrated `predict_proba` automatically once retrained with `SVC(probability=True)`.

> A rendered PDF version of this diagram is in [PIPELINE.pdf](PIPELINE.pdf).
