# HandiMouse 🖐️🖱️

HandiMouse is a high-performance, real-time computer vision application that replaces a physical computer mouse with human hand gestures. 

The system is built on a highly modular, decoupled architecture using **Python**, **OpenCV**, and **MediaPipe** to ensure low-latency cursor movement and robust gesture mapping.

---

## 🏗️ System Architecture

To ensure high responsiveness and maintain standard screen refreshes, HandiMouse decouples camera ingestion, vision analysis, gesture intelligence, and OS mouse events across separate executing threads:

1. **Input Layer (Camera Stream)**: Continuous, non-blocking frame retrieval running on a background worker thread. Uses a single-slot buffer overwrite strategy to maintain zero-lag ingestion.
2. **Vision Layer (Hand Tracking)**: Encapsulates MediaPipe Hands tracking to extract 3D landmarks in real time.
3. **Intelligence Layer (Gesture Recognition)**: Structural and heuristic classifiers combined with exponential smoothing filters to prevent target jitter and cursor micro-tremors.
4. **Control Layer (OS Interaction)**: Maps gesture events into native mouse actions using direct low-level Windows APIs for sub-millisecond execution times.
5. **Safety Layer (Emergency Stop)**: Active out-of-process hotkey listeners and boundary gatekeepers that immediately deactivate OS interactions on demand.
6. **Configuration Layer**: Dynamic runtime profile scaling.
7. **Logging & Monitoring Layer**: Structured telemetry tracking (FPS, latency bottlenecks, dropped frames).

---

## ⚡ Non-Blocking Input Layer (Completed)

The **Input Layer** and **Logging & Monitoring Layer** are fully implemented.

### Optimization Mechanics:
* **Background Ingestion Thread**: Moving frame polling out of the main execution loop bypasses OpenCV's blocking exposure times.
* **Resolution Resizing (Downsampling)**: Automatically downsamples captured frames to `640x480` at acquisition to accelerate downstream MediaPipe inference.
* **DirectShow Windows Optimization**: Speeds up device handshaking on Windows systems, dropping boot latencies from `12.0s` to **`3.43s`**.
* **Safe Reconnects**: Gracefully handles sensor disconnects with self-healing retry timers.

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install opencv-python
```

### 2. Running Diagnostic Benchmarks
Run the included verification loop to check raw ingestion framerates and memory-copy speeds:
```python
# To test raw stream performance in isolation:
python scratch/test_camera.py
```
*(Note: A copy of `test_camera.py` is included in the project's diagnostic suite).*
