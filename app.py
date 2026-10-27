
import os
import re
import time
import tempfile
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
import streamlit as st

st.set_page_config(
    page_title="ISL Word Recognizer | ML Mini Project",
    page_icon="🤟",
    layout="wide"
)

# --- Constants & MediaPipe Setup ---
NUM_FRAMES = 24
POSE_INDICES = [11, 12, 13, 14, 15, 16]
MODEL_BUNDLE_PATH = Path("isl_word_recognizer_final.joblib")
DATA_DIR = Path("dataset_extracted")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

@st.cache_resource
def load_model_bundle():
    if not MODEL_BUNDLE_PATH.exists():
        st.error(f"Model file `{MODEL_BUNDLE_PATH}` not found! Run Cell 11 in the notebook first.")
        st.stop()
    return joblib.load(MODEL_BUNDLE_PATH)

bundle = load_model_bundle()
model = bundle["model"]
model_name = bundle.get("model_name", "SVM (RBF)")
le = bundle["label_encoder"]
target_words = bundle["target_words"]

# --- Core Feature Extraction & Visualization ---
def process_video_and_preview(video_path: str, num_frames: int = NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None, 0.0, []

    frame_idxs = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
    seq = np.full((num_frames, 270), np.nan, dtype=np.float32)
    hand_detected_frames = 0
    preview_frames = []
    preview_indices = set(np.linspace(0, num_frames - 1, 6, dtype=int))

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        for i, f_idx in enumerate(frame_idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            if w > 640:
                scale = 640.0 / w
                frame = cv2.resize(frame, (640, int(h * scale)))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            # Shoulder-midpoint & shoulder-width normalization
            if results.pose_landmarks:
                plm = results.pose_landmarks.landmark
                l_sh = np.array([plm[11].x, plm[11].y, plm[11].z], dtype=np.float32)
                r_sh = np.array([plm[12].x, plm[12].y, plm[12].z], dtype=np.float32)
                origin = 0.5 * (l_sh + r_sh)
                sh_width = np.linalg.norm(l_sh[:2] - r_sh[:2])
                if sh_width < 1e-4:
                    sh_width = 1.0

                pose_feats = []
                for idx in POSE_INDICES:
                    pt = np.array([plm[idx].x, plm[idx].y, plm[idx].z], dtype=np.float32)
                    pose_feats.extend((pt - origin) / sh_width)
                seq[i, 0:18] = pose_feats
            else:
                origin = np.array([0.5, 0.5, 0.0], dtype=np.float32)
                sh_width = 1.0

            has_hand = False
            if results.left_hand_landmarks:
                has_hand = True
                lhl = results.left_hand_landmarks.landmark
                pts = np.array([[lm.x, lm.y, lm.z] for lm in lhl], dtype=np.float32)
                wrist = pts[0].copy()
                seq[i, 18:81] = ((pts - origin) / sh_width).flatten()
                seq[i, 81:144] = ((pts - wrist) / sh_width).flatten()

            if results.right_hand_landmarks:
                has_hand = True
                rhl = results.right_hand_landmarks.landmark
                pts = np.array([[lm.x, lm.y, lm.z] for lm in rhl], dtype=np.float32)
                wrist = pts[0].copy()
                seq[i, 144:207] = ((pts - origin) / sh_width).flatten()
                seq[i, 207:270] = ((pts - wrist) / sh_width).flatten()

            if has_hand:
                hand_detected_frames += 1

            # Draw landmarks on 6 evenly spaced key-frames for UI visualization
            if i in preview_indices:
                annotated = rgb.copy()
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
                    )
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
                    )
                preview_frames.append((f"Frame {i+1}/{num_frames}", annotated))

    cap.release()
    det_rate = hand_detected_frames / float(num_frames)
    return seq, det_rate, preview_frames

def sequence_to_feature_row(seq: np.ndarray):
    df_seq = pd.DataFrame(seq)
    nan_count = int(df_seq.isna().sum().sum())
    df_interp = df_seq.interpolate(method='linear', axis=0, limit_direction='both').fillna(0.0)
    arr = df_interp.to_numpy(dtype=np.float32)

    f_mean = np.mean(arr, axis=0)
    f_std  = np.std(arr, axis=0)
    f_min  = np.min(arr, axis=0)
    f_max  = np.max(arr, axis=0)
    f_vel  = np.mean(np.abs(np.diff(arr, axis=0)), axis=0)

    return np.concatenate([f_mean, f_std, f_min, f_max, f_vel]), nan_count

def run_prediction_ui(video_path: str, ground_truth: str = None):
    with st.spinner("Extracting MediaPipe Holistic landmarks & running SVM (RBF) pipeline..."):
        t0 = time.time()
        seq, det_rate, preview_frames = process_video_and_preview(video_path)
        if seq is None:
            st.error("Could not decode video frames.")
            return
        feat_row, nan_count = sequence_to_feature_row(seq)
        X_single = feat_row.reshape(1, -1)

        pred_idx = model.predict(X_single)[0]
        pred_word = le.inverse_transform([pred_idx])[0]
        proba = model.predict_proba(X_single)[0]
        elapsed = time.time() - t0

    top5_idxs = np.argsort(proba)[::-1][:5]
    top5_words = [le.inverse_transform([i])[0] for i in top5_idxs]
    top5_probs = [float(proba[i]) * 100 for i in top5_idxs]

    # --- Metrics Row ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted ISL Sign", pred_word.upper(), f"{top5_probs[0]:.1f}% Confidence")
    if ground_truth:
        match = "✅ MATCH" if pred_word.lower() == ground_truth.lower() else "❌ MISMATCH"
        c2.metric("Ground Truth Label", ground_truth.upper(), match)
    else:
        c2.metric("Model Pipeline", f"Tuned {model_name}", "1,350 Features")
    c3.metric("Hand Detection Rate", f"{det_rate*100:.1f}%", f"{int(det_rate*24)}/24 Frames")
    c4.metric("Inference Latency", f"{elapsed:.2f} sec", f"{nan_count} NaNs imputed")

    st.divider()

    # --- Visual Layout: Left = Keyframes with Landmarks, Right = Top-5 Probability Chart ---
    col_left, col_right = st.columns([1.35, 1.0])

    with col_left:
        st.subheader("🦴 Extracted MediaPipe Skeletal Keyframes (Normalized)")
        if preview_frames:
            grid_cols = st.columns(3)
            for idx, (caption, img) in enumerate(preview_frames):
                with grid_cols[idx % 3]:
                    st.image(img, caption=caption, use_container_width=True)

    with col_right:
        st.subheader("📊 Top-5 Class Probabilities (SVM RBF)")
        df_probs = pd.DataFrame({
            "ISL Word": [w.upper() for w in top5_words],
            "Probability (%)": [round(p, 2) for p in top5_probs]
        })
        st.bar_chart(df_probs.set_index("ISL Word"), horizontal=True, height=260)
        st.dataframe(df_probs, use_container_width=True, hide_index=True)

# --- Sidebar Info ---
with st.sidebar:
    st.title("🤟 ISL Word Recognizer")
    st.caption("Sem V INFT — Machine Learning Mini Project (PCIT15P)")
    st.markdown(f"""
    **Final Pipeline:**
    - `StandardScaler`
    - `PCA(n_components=None)`
    - `SVC(kernel='rbf', C=5.0, gamma=0.001)`
    
    **Benchmark Metrics:**
    - **5-Fold CV Macro-F1:** `93.14%`
    - **Holdout Test Accuracy:** `100.0%`
    - **Dataset:** AI4Bharat INCLUDE (209 clips, 13 classes)
    """)
    st.markdown("**Supported 13 Classes:**")
    st.code(", ".join(target_words), language="text")

# --- Main Header & Mode Tabs ---
st.title("🤟 Indian Sign Language (ISL) Isolated Word Recognition")
st.markdown("Classical Machine Learning Pipeline using **MediaPipe Holistic Landmarks (270 coords × 5 temporal stats = 1,350 features)** + **Tuned RBF-SVM**.")

tab1, tab2, tab3 = st.tabs([
    "📂 1. Test on Dataset Videos (Recommended for Viva)",
    "📤 2. Upload Any Video (.MOV / .MP4)",
    "📷 3. Live Webcam 3-Sec Recorder"
])

# --- TAB 1: Dataset Explorer ---
with tab1:
    @st.cache_data
    def scan_dataset_videos():
        records = []
        if DATA_DIR.exists():
            for root, _, files in os.walk(DATA_DIR):
                if Path(root).name == "Extra":
                    continue
                for f in sorted(files):
                    if f.lower().endswith(('.mov', '.mp4')) and not f.startswith('._'):
                        raw_folder = Path(root).name
                        label = re.sub(r'^\d+\.\s*', '', raw_folder.strip()).lower()
                        label = re.sub(r'\s+', ' ', label)
                        if label in target_words:
                            records.append({"label": label, "filename": f, "path": str(Path(root) / f)})
        return pd.DataFrame(records)

    df_vids = scan_dataset_videos()
    if df_vids.empty:
        st.warning("No dataset videos found inside `dataset_extracted/`. Use Tab 2 to upload a video.")
    else:
        sc1, sc2, sc3 = st.columns([1, 1, 0.8])
        with sc1:
            selected_word = st.selectbox("Select Ground-Truth ISL Word Class:", sorted(df_vids["label"].unique()))
        subset = df_vids[df_vids["label"] == selected_word]
        with sc2:
            selected_file = st.selectbox("Select Video Clip (.MOV):", subset["filename"].tolist())
        selected_path = subset[subset["filename"] == selected_file]["path"].values[0]
        with sc3:
            st.write("")
            st.write("")
            run_btn = st.button("🚀 Run ISL Prediction", type="primary", use_container_width=True)

        if run_btn:
            run_prediction_ui(selected_path, ground_truth=selected_word)

# --- TAB 2: Upload Video ---
with tab2:
    uploaded = st.file_uploader("Upload an isolated ISL sign video (.MOV, .MP4, .AVI)", type=["mov", "mp4", "avi"])
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        if st.button("🔍 Classify Uploaded Video", type="primary"):
            run_prediction_ui(tmp_path)

# --- TAB 3: Live Webcam Recorder ---
with tab3:
    st.info("💡 Sit with your upper body and shoulders clearly visible in frame. Click the button below to record a **3-second sign clip** from your laptop webcam.")
    if st.button("🎥 Start 3-Second Webcam Sign Capture", type="primary"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access webcam (camera index 0).")
        else:
            frame_placeholder = st.empty()
            progress_bar = st.progress(0, text="Get ready... Recording 3 seconds of signing!")
            frames_captured = []
            fps = 20
            total_to_capture = fps * 3  # ~60 frames over 3 seconds

            for i in range(total_to_capture):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_flipped = cv2.flip(frame, 1)
                frames_captured.append(frame)  # Keep unflipped original orientation matching DSLR dataset
                frame_placeholder.image(cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB), caption=f"Recording frame {i+1}/{total_to_capture}...", width=480)
                progress_bar.progress((i + 1) / total_to_capture, text=f"Recording... {int((i+1)/total_to_capture*100)}%")
                time.sleep(1.0 / fps)

            cap.release()
            frame_placeholder.empty()
            progress_bar.empty()

            if frames_captured:
                h, w = frames_captured[0].shape[:2]
                tmp_webcam = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(tmp_webcam, fourcc, fps, (w, h))
                for f in frames_captured:
                    out.write(f)
                out.release()
                st.success("✅ 3-second sign captured! Running MediaPipe + SVM pipeline...")
                run_prediction_ui(tmp_webcam)