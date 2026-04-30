import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
import time
import json
from pathlib import Path


from tkinter import Tk
from tkinter.filedialog import askopenfilename
import sys


CHUNK_DURATION = 5  # seconds


# ------------------------------
# Select Video File
# ------------------------------
def select_video():
    root = Tk()
    root.withdraw()

    video_path = askopenfilename(
        title="Select Face Video",
        filetypes=[
            ("Video Files", "*.mp4 *.avi *.mov *.mkv")
        ]
    )

    if not video_path:
        print("No video selected.")
        sys.exit()

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
    peak_power = np.max(yf[valid_idx])
    avg_power = np.mean(yf[valid_idx])

    confidence = peak_power / avg_power

    confidence = min(round(confidence / 10, 2), 1.0)

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
# Run
# ------------------------------
if __name__ == "__main__":

    try:
        VIDEO_PATH = select_video()

        print(f"Selected Video: {VIDEO_PATH}")

        results = process_video(VIDEO_PATH)

        Path("output").mkdir(exist_ok=True)

        with open("output/results.json", "w") as f:
            json.dump(results, f, indent=2)

        print("\nFinal Results:\n")
        print(json.dumps(results, indent=2))

    except Exception as e:
        print("\nProcessing Stopped")
        print(f"Reason: {e}")