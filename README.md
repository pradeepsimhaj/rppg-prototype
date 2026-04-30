# Near Real-Time rPPG Prototype

## Overview

This project is a basic near real-time remote photoplethysmography (rPPG) prototype that estimates physiological biomarkers from a facial video using computer vision and signal processing techniques.

The application processes a facial video incrementally in 5-second chunks and estimates:

- Heart Rate (BPM)
- Respiratory Rate
- HRV (Heart Rate Variability)
- Stress Level
- Confidence Score
- Runtime / Performance Metrics

The application is implemented using Python, OpenCV, NumPy, SciPy, and Streamlit.

---

# Features

## Core Features

- Upload facial video through Streamlit UI
- Incremental 5-second chunk processing
- BPM estimation per chunk
- Overall BPM aggregation
- Respiratory Rate estimation
- HRV estimation
- Stress level estimation
- Confidence score calculation
- Runtime and latency metrics
- Graphical visualization of biomarker trends
- JSON export support

---

# Tech Stack

## Frontend
- Streamlit

## Computer Vision
- OpenCV Haar Cascade Face Detection

## Signal Processing
- NumPy
- SciPy FFT
- Bandpass Filtering

## Visualization
- Pandas
- Matplotlib

---

# Application Workflow

## 1. Video Upload

The user uploads a facial video through the Streamlit interface.

Supported formats:
- MP4
- AVI
- MOV
- MKV

---

## 2. Video Validation

The system validates:
- Video duration must be between 50–70 seconds
- Only one face should be visible
- Face should remain visible throughout processing

Processing stops automatically if:
- No face is detected
- Multiple faces are detected
- Face visibility becomes unstable

---

## 3. Chunk-Based Processing

The uploaded video is divided into:
- 5-second chunks

Each chunk undergoes:
- Face detection
- ROI extraction (forehead region)
- Green channel signal extraction
- Signal filtering
- FFT-based frequency analysis

---

## 4. Biomarker Estimation

### BPM Estimation
Heart rate is estimated using FFT peak frequency analysis within physiological heart rate frequency bands.

### Respiratory Rate
Respiratory rate is estimated using low-frequency signal variations.

### HRV
HRV is approximated using temporal pulse variability.

### Stress Level
Stress level is heuristically estimated from HRV values.

### Confidence Score
Confidence score estimates signal quality and prediction reliability.

---

# Performance Metrics

The application measures:
- Total runtime
- Chunk processing runtime
- Average chunk latency
- Number of processed chunks

---

# Graphical Visualization

The application visualizes:
- BPM trend
- Respiratory rate trend
- HRV trend
- Confidence score trend

using:
- Pandas DataFrames
- Matplotlib charts

---

# Project Structure

```bash
rppg-prototype/
│
├── streamlit_app.py
├── requirements.txt
├── output/
│   └── results.json
└── README.md

git clone https://github.com/pradeepsimhaj/rppg-prototype

cd rppg-prototype

pip install -r requirements.txt

streamlit run streamlit_app.py


# Deployment

The application can be deployed on:

* Streamlit Cloud
* HuggingFace Spaces
* Render
* Railway

---

# Important Deployment Note

Use:

```txt
opencv-python-headless
```

instead of:

```txt
opencv-python
```

to avoid deployment issues in cloud environments.

---

# Example Output

## Overall Metrics

| Metric           | Value    |
| ---------------- | -------- |
| BPM              | 76       |
| Respiratory Rate | 16       |
| HRV              | 54 ms    |
| Stress Level     | Moderate |
| Confidence Score | 0.72     |

---

## Sample Chunk Output

| Chunk | BPM | RR | HRV | Confidence |
| ----- | --- | -- | --- | ---------- |
| 0     | 74  | 15 | 52  | 0.68       |
| 1     | 76  | 16 | 55  | 0.71       |
| 2     | 78  | 17 | 58  | 0.74       |

---

# Known Limitations

This is a prototype implementation and has several limitations:

* Not medically validated
* Sensitive to lighting conditions
* Sensitive to motion artifacts
* Reduced accuracy with low FPS videos
* Approximate HRV and stress estimation
* FFT-based approach is simplistic compared to deep learning rPPG models

---

# Future Improvements

Potential improvements include:

* Deep learning based rPPG models
* Better ROI tracking
* Real-time webcam support
* GPU acceleration
* Better HRV estimation
* More robust respiratory signal extraction
* Multi-face handling
* Temporal smoothing

---

# AI Usage Disclosure

AI tools were used to assist with:

* Architecture planning
* Signal processing integration
* Streamlit UI development
* Debugging deployment issues
* Documentation generation
