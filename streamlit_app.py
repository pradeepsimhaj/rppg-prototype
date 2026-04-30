import streamlit as st
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
import time
import json
from pathlib import Path
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import sys


CHUNK_DURATION = 5  # seconds


# ------------------------------
# Select Video File
# ------------------------------
def select_video():

    video_path = st.file_uploader(
        title="Select Face Video",
        filetypes=[
            ("Video Files", "*.mp4 *.avi *.mov *.mkv")
        ]
    )

    return video_path

# ------------------------------
# OpenCV Haar Cascade Face Detector
# ------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)


# ------------------------------
# Bandpass Filter
# ------------------------------
def bandpass_filter(signal, fs, low=0.8, high=3.0):
    nyquist = 0.5 * fs
    low /= nyquist
    high /= nyquist

    b, a = butter(3, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered


# ------------------------------
# Respiratory Filter
# ------------------------------
def respiratory_filter(signal, fs, low=0.1, high=0.5):
    nyquist = 0.5 * fs
    low /= nyquist
    high /= nyquist

    b, a = butter(2, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)

    return filtered



# ------------------------------
# BPM Estimation
# ------------------------------
def estimate_bpm(signal, fs):

    if len(signal) < fs * 2:
        return None, None

    signal = signal - np.mean(signal)

    try:
        filtered = bandpass_filter(signal, fs)
    except:
        return None, None

    n = len(filtered)

    yf = np.abs(fft(filtered))
    xf = fftfreq(n, 1 / fs)

    valid_idx = np.where((xf >= 0.8) & (xf <= 3.0))

    if len(valid_idx[0]) == 0:
        return None, None

    peak_freq = xf[valid_idx][np.argmax(yf[valid_idx])]

    bpm = peak_freq * 60

    # Confidence Score
    # Confidence Score
    peak_power = np.max(yf[valid_idx])

    mean_power = np.mean(yf[valid_idx])

    ratio = peak_power / mean_power

    confidence = ratio / (ratio + 2)

    confidence = round(confidence, 2)

    return round(float(bpm), 2), confidence


# ------------------------------
# Respiratory Rate Estimation
# ------------------------------
def estimate_respiratory_rate(signal, fs):

    if len(signal) < fs * 2:
        return None

    signal = signal - np.mean(signal)

    try:
        filtered = respiratory_filter(signal, fs)
    except:
        return None

    n = len(filtered)

    yf = np.abs(fft(filtered))
    xf = fftfreq(n, 1 / fs)

    valid_idx = np.where((xf >= 0.1) & (xf <= 0.5))

    if len(valid_idx[0]) == 0:
        return None

    peak_freq = xf[valid_idx][np.argmax(yf[valid_idx])]

    rr = peak_freq * 60

    return round(float(rr), 2)



# ------------------------------
# HRV Estimation
# ------------------------------
def estimate_hrv(signal, fs):

    try:
        filtered = bandpass_filter(signal, fs)

        peaks, _ = find_peaks(filtered, distance=fs/2)

        if len(peaks) < 2:
            return None

        peak_intervals = np.diff(peaks) / fs

        hrv = np.std(peak_intervals) * 1000

        return round(float(hrv), 2)

    except:
        return None


# ------------------------------
# Stress Estimation
# ------------------------------
def estimate_stress(hrv):

    if hrv is None:
        return "Unknown"

    if hrv > 80:
        return "Low"

    elif hrv > 40:
        return "Moderate"

    else:
        return "High"



# ------------------------------
# Extract ROI Signal
# ------------------------------
def extract_green_signal(frames, fps):
    signal = []

    for frame_index, frame in enumerate(frames):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )

        # ------------------------------
        # Validation Checks
        # ------------------------------

        # No face
        if len(faces) == 0:
            raise Exception(
                f"No face detected at frame {frame_index}"
            )

        # More than one face
        if len(faces) > 1:
            raise Exception(
                f"Multiple faces detected at frame {frame_index}"
            )

        # Single face
        x, y, w, h = faces[0]

        # Forehead ROI
        roi_y1 = y
        roi_y2 = y + h // 4

        roi_x1 = x + w // 4
        roi_x2 = x + (3 * w // 4)

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        if roi.size == 0:
            continue

        green_mean = np.mean(roi[:, :, 1])

        signal.append(green_mean)

    return np.array(signal)



# ------------------------------
# Main Pipeline
# ------------------------------
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("Unable to open video")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"FPS: {fps}")
    print(f"Duration: {duration:.2f} sec")

    chunk_frames = fps * CHUNK_DURATION

    chunk_results = []
    all_bpms = []
    all_rr = []
    all_hrv = []
    all_confidence = []


    chunk_index = 0

    start_total = time.time()
    
    while True:
        frames = []

        for _ in range(chunk_frames):
            ret, frame = cap.read()

            if not ret:
                break

            frames.append(frame)

        if len(frames) == 0:
            break

        start_chunk = time.time()

        signal = extract_green_signal(frames, fps)

        bpm, confidence = estimate_bpm(signal, fps)

        respiratory_rate = estimate_respiratory_rate(signal, fps)

        hrv = estimate_hrv(signal, fps)
        
        stress_level = estimate_stress(hrv)

        chunk_runtime = time.time() - start_chunk


        chunk_info = {
            "chunk": chunk_index,
            "start_sec": chunk_index * CHUNK_DURATION,
            "end_sec": (chunk_index + 1) * CHUNK_DURATION,
            "bpm": bpm,
            "respiratory_rate": respiratory_rate,
            "hrv_ms": hrv,
            "stress_level": stress_level,
            "confidence_score": confidence,
            "frames_processed": len(frames),
            "runtime_sec": round(chunk_runtime, 3)
        }

        chunk_results.append(chunk_info)

        if bpm:
            all_bpms.append(bpm)

        if respiratory_rate:
            all_rr.append(respiratory_rate)

        if hrv:
            all_hrv.append(hrv)

        if confidence:
            all_confidence.append(confidence)

        print(chunk_info)

        chunk_index += 1

    cap.release()

    total_runtime = time.time() - start_total
    
    overall_bpm = round(np.mean(all_bpms), 2) if all_bpms else None

    overall_rr = round(np.mean(all_rr), 2) if all_rr else None

    overall_hrv = round(np.mean(all_hrv), 2) if all_hrv else None

    overall_confidence = round(np.mean(all_confidence), 2) if all_confidence else None

    overall_stress = estimate_stress(overall_hrv)


    final_results = {
        "overall_metrics": {
            "overall_bpm": overall_bpm,
            "overall_respiratory_rate": overall_rr,
            "overall_hrv_ms": overall_hrv,
            "overall_stress_level": overall_stress,
            "overall_confidence_score": overall_confidence
        },
        "chunks": chunk_results,
        "performance_metrics": {
            "video_duration_sec": round(duration, 2),
            "total_runtime_sec": round(total_runtime, 2),
            "avg_chunk_runtime_sec": round(
                np.mean([c['runtime_sec'] for c in chunk_results]),
                3
            ) if chunk_results else 0,
            "total_chunks": len(chunk_results),
            "fps": fps,
            "total_frames": total_frames
        }
    }

    return final_results


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(
    page_title="Near Real-Time rPPG",
    layout="wide"
)

st.title("Near Real-Time rPPG Prototype")

st.markdown("""
This prototype performs remote photoplethysmography (rPPG) analysis
from facial videos using computer vision and signal processing.

The pipeline processes the uploaded video in 5-second chunks and estimates:
- Heart Rate (BPM)
- Respiratory Rate
- HRV (Heart Rate Variability)
- Stress Level
- Confidence Score
- Runtime Metrics
""")

# ------------------------------
# Video Requirements
# ------------------------------
st.subheader("Video Upload Requirements")

st.info("""
Please ensure the following before uploading the video:

### Video Constraints
- Only ONE video can be processed at a time
- Recommended video duration: 60 seconds
- Supported formats:
  - MP4
  - AVI
  - MOV
  - MKV

### Face Visibility Requirements
- Exactly ONE face should be visible throughout the video
- Face should remain clearly visible
- Frontal face orientation recommended
- Avoid extreme head movement
- Avoid occlusions (mask, hand, sunglasses, etc.)
- Good lighting conditions recommended

### Processing Validation
The system automatically stops processing if:
- No face is detected
- Multiple faces are detected
- Face visibility becomes unstable

### Biomarker Outputs
The prototype estimates:
- BPM (Heart Rate)
- Respiratory Rate
- HRV
- Stress Level
- Confidence Score

### Disclaimer
This implementation is intended for research and prototype demonstration only.
It is NOT medically validated.
""")

# ------------------------------
# Biomarker Details
# ------------------------------
st.subheader("Estimated Biomarkers")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
### Cardiovascular Metrics
- **BPM (Heart Rate)**  
  Estimated using forehead color signal variations.

- **HRV (Heart Rate Variability)**  
  Estimated from temporal pulse interval variations.

- **Confidence Score**  
  Indicates signal quality and estimation stability.
""")

with col2:
    st.markdown("""
### Wellness Metrics
- **Respiratory Rate**  
  Estimated from low-frequency physiological variations.

- **Stress Level**  
  Derived heuristically using HRV patterns.

- **Runtime Metrics**  
  Measures chunk-wise and total pipeline latency.
""")

# ------------------------------
# Upload Section
# ------------------------------
st.subheader("Upload Face Video")

uploaded_video = st.file_uploader(
    "Upload a single facial video",
    type=["mp4", "avi", "mov", "mkv"],
    accept_multiple_files=False
)

if uploaded_video:

    st.video(uploaded_video)

    temp_file = tempfile.NamedTemporaryFile(delete=False)

    temp_file.write(uploaded_video.read())

    temp_path = temp_file.name

    # ------------------------------
    # Validate Video Duration
    # ------------------------------
    cap = cv2.VideoCapture(temp_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = total_frames / fps

    cap.release()

    st.subheader("Uploaded Video Details")

    col1, col2, col3 = st.columns(3)

    col1.metric("FPS", fps)
    col2.metric("Duration (sec)", round(duration, 2))
    col3.metric("Total Frames", total_frames)

    # ------------------------------
    # Duration Validation
    # ------------------------------
    if duration < 50 or duration > 70:

        st.error(
            "Invalid video duration. "
            "Video length must be between 50 and 70 seconds."
        )

        st.stop()

    else:

        st.success(
            "Video duration validation passed "
            "(50–70 seconds accepted)"
        )

# ------------------------------
# Process Button
# ------------------------------

    if st.button("Process Video"):

        try:

            with st.spinner("Processing video..."):

                results = process_video(temp_path)

            st.success("Processing Complete")

            overall = results["overall_metrics"]

            st.subheader("Overall Metrics")

            col1, col2, col3 = st.columns(3)

            col1.metric("BPM", overall["overall_bpm"])
            col2.metric("Respiratory Rate", overall["overall_respiratory_rate"])
            col3.metric("HRV", overall["overall_hrv_ms"])

            col4, col5 = st.columns(2)

            col4.metric("Stress Level", overall["overall_stress_level"])
            col5.metric("Confidence", overall["overall_confidence_score"])

            st.subheader("Chunk-wise Results")

            # Convert JSON to DataFrame
            df = pd.DataFrame(results["chunks"])

            # Display table
            st.dataframe(df, use_container_width=True)

            # ------------------------------
            # BPM Graph
            # ------------------------------
            st.subheader("BPM Trend")

            fig1, ax1 = plt.subplots()

            ax1.plot(
                df["chunk"],
                df["bpm"],
                marker='o'
            )

            ax1.set_xlabel("Chunk")
            ax1.set_ylabel("BPM")
            ax1.set_title("BPM Over Time")

            st.pyplot(fig1)

            # ------------------------------
            # Respiratory Rate Graph
            # ------------------------------
            st.subheader("Respiratory Rate Trend")

            fig2, ax2 = plt.subplots()

            ax2.plot(
                df["chunk"],
                df["respiratory_rate"],
                marker='o'
            )

            ax2.set_xlabel("Chunk")
            ax2.set_ylabel("Respiratory Rate")
            ax2.set_title("Respiratory Rate Over Time")

            st.pyplot(fig2)

            # ------------------------------
            # HRV Graph
            # ------------------------------
            st.subheader("HRV Trend")

            fig3, ax3 = plt.subplots()

            ax3.plot(
                df["chunk"],
                df["hrv_ms"],
                marker='o'
            )

            ax3.set_xlabel("Chunk")
            ax3.set_ylabel("HRV (ms)")
            ax3.set_title("HRV Over Time")

            st.pyplot(fig3)

            # ------------------------------
            # Confidence Score Graph
            # ------------------------------
            st.subheader("Confidence Score Trend")

            fig4, ax4 = plt.subplots()

            ax4.plot(
                df["chunk"],
                df["confidence_score"],
                marker='o'
            )

            ax4.set_xlabel("Chunk")
            ax4.set_ylabel("Confidence")
            ax4.set_title("Confidence Score Over Time")

            st.pyplot(fig4)


            st.subheader("Performance Metrics")

            performance_df = pd.DataFrame(
                [results["performance_metrics"]]
            )

            st.dataframe(
                performance_df,
                use_container_width=True
            )

            Path("output").mkdir(exist_ok=True)

            with open("output/results.json", "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:

            st.error(f"Processing Failed: {e}")