
# 🚁 FireDrone-X Final Project – ENAE788M

### 👨‍🔬 Team Members
- **Sai Jagadeesh Muralikrishnan** (120172243)
- **Varun Lakshmanan** (120169595)

---

## 📌 Project Overview

**FireDrone-X** is an autonomous VOXL2-based drone system that detects **fire** and **humans** in real-time to aid in emergency response operations. It utilizes onboard cameras (stereo, high-res, tracking), GPS modules, and ROS 2 nodes for detection, trajectory planning, and GUI-based mission reporting.

We are using **open-source fire and human detection models** (e.g., YOLOv8 for humans and pretrained fire detection CNNs) instead of building a CNN from scratch with FIRESENSE.

---

## 🔧 Hardware Setup

- **Platform**: VOXL2 Flight Deck (MDK-F0006-4-V1-C11)
- **Cameras**: Stereo + Hi-res + Tracking (Cam Config 11)
- **Flight Control**: VOXL2 IO PWM ESC, M0065 SBUS, GPS + EKF2
- **GUI Interface**: PyQt5

---

## 🧠 Software Architecture

### ROS 2 Packages:
```
fire_detector/         # OpenCV + pretrained fire detection CNN
human_detector/        # YOLOv8 (ONNX Runtime) for person detection
mission_control/       # Fire zone scoring, GPS path planning
gui/                   # PyQt5 GUI to display live data and export reports
scripts/voxl_interface_node.py  # Camera and GPS integration
```

---

## 📅 Timeline & Milestones

### ✅ Phase 1: Simulation Setup (Due **May 11**)
- [ ] Build simulation environment in Unreal Engine or Gazebo
- [ ] Integrate person detection (YOLOv8)
- [ ] Integrate fire detection using open-source model
- [ ] Simulate GPS and fire values
- [ ] Connect to GUI base station and verify telemetry updates

### ✅ Phase 2: Real Drone Setup & Pre-Report (Due **May 16**)
- [ ] Set up VOXL2 and calibrate cameras
- [ ] Define ROS 2 package dependencies to replicate simulation
- [ ] Finalize Python file structure for detection and mission nodes
- [ ] Run detection-only tests (no flight)
- [ ] Save test results and log ROS 2 bags

### ✅ Phase 3: Final Report (Due **May 18**, buffer till May 20)
- [ ] Include all detection results and bag logs
- [ ] MATLAB-based graphs and analysis (severity, count, locations)
- [ ] Add republisher node details if used
- [ ] Compile results and export JSON/CSV reports from GUI

---

## 📦 Dataset & Tools

- 🔥 Fire Models: Pretrained fire detection models (open-source)
- 🧍 Person Detection: YOLOv8 + ONNX Runtime
- 📷 Datasets: COCO, VisDrone
- 💻 Tools: VOXL SDK, QGroundControl, MATLAB, PyQt5, ROS 2 Humble

---

## 📝 Notes

- [ ] Capture ROS 2 bag files for **every test and real-world run**
- [ ] Add republisher node if topic alignment is required
- [ ] Maintain logs and backups of model inferences and GUI exports
- [ ] Keep GUI lightweight and telemetry intuitive for field use

---

## 🔗 References

1. [VOXL2 SDK](https://docs.modalai.com)
2. [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
3. [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
4. [ONNX Runtime](https://onnxruntime.ai/)
5. [PyQt5 GUI Reference](https://build-system.fman.io/pyqt5-tutorial)
