# FireDrone-X: Autonomous UAV Fire & Human Detection System

## Team Members
- **Sai Jagadeesh Muralikrishnan** (120172243)
- **Varun Lakshmanan** (120169595)

---

## Overview

**FireDrone-X** is an autonomous drone system powered by VOXL2 that detects **fire** (using cones as proxies) and **humans** in real-time using a monocular RGB camera. It integrates **depth estimation**, **trajectory planning**, and **fire tracking logic** using ROS 2 for effective mission handling in fire hazard zones.

Two main fire localization strategies were implemented:
- **Monocular Projection**: Estimates fire positions using pixel-to-ground transformation with known camera intrinsics and pitch.
- **Monocular Depth Estimation**: Uses the **DepthAnythingV2** model to extract true 3D (x, y, z) coordinates directly from the depth map.

The system supports **live plotting**, **RTSP video streaming**, and a lightweight **PyQt5 GUI** for monitoring.

---

## Software Modules

### Detection Modules
- `yolo_firedronex.py`: YOLOv8-based fire/person detection with projected localization.
- `yolo_firedronex_real.py`: Optimized version for VOXL2 onboard runtime.

### Mission Planning Modules
- `firedronex_depth.py`: Mission logic using monocular depth estimation for getting the position of cone.
- `firedronex_depth_real.py`: Real-time version of Monacular depth estimation method with VOXL2-specific optimizations.
- `moncular_projection_firedroneX.py`: Mission logic using monocular projection which is based on trignometry.

### Visualization & Streaming
- `firedronex_live_plotter.py`: Real-time fire/person detection plotting.
- `rtsp_stream_publisher.py`: Used to publish camera stream to GUI or external RTSP viewers.

### Depth Estimation
- `depth_anything_v2_estimator.py`: Runs the DepthAnythingV2 model to extract dense depth maps for using those in Monacular Depth Estimation of 3D localization.

### GUI Interface
- `gui.py`: PyQt5-based GUI for mission state visualization and exporting reports.

---

## Datasets & Models

- **Fire & Person Detection**: YOLOv8 (pretrained on custom + COCO datasets)
- **Monocular Depth**: DepthAnythingV2 (pretrained weights)
- **Sample Datasets**: COCO, VisDrone, custom simulation cone/person sets

---

## Design Workflow

The design of **FireDrone-X** was centered around modularity, robustness, and compatibility with the VOXL2 platform. The overall workflow is broken down into four primary stages:

---

### 1. **System Architecture Design**

- **Objective**: Real-time detection and localization of fire and humans from a UAV using a monocular camera.
- **Constraints**:
  - Edge inference on VOXL2 (limited compute)
  - ROS 2 Humble compatibility
  - Monocular setup (no stereo/depth sensor)
- **Design Choices**:
  - Use YOLOv8 for fire and person detection due to its accuracy and lightweight deployment.
  - Integrate **DepthAnythingV2** to extract depth maps from a single RGB image to support 3D localization.
  - Implement modular mission planning logic supporting two strategies: monocular projection and monocular depth.

---

### 2. **Module Development & Integration**

| Module | Description |
|--------|-------------|
| **YOLOv8 Detection** | Fire and human bounding box detection from RGB stream |
| **Depth Estimation** | Generates per-pixel depth using DepthAnythingV2 |
| **Localization Logic** | Estimates (x, y, z) using either trigonometry or dense depth |
| **Trajectory Planner** | Selects targets and sends PX4-compatible setpoints |
| **GUI + Visualization** | Plots detection, state transitions, and publishes RTSP stream |
| **ROS 2 Messaging** | Custom messages for detection, localization, and setpoint handling |

- All modules are containerized and optimized for VOXL2.
- A fallback mechanism allows switching between projection and depth-based localization at runtime.

---

### 3. **Mission Execution Flow**

```text
+------------------+
|  RGB Camera Feed |
+--------+---------+
         |
         v
+--------+---------+           +-------------------------+
|   YOLOv8 Detector  | ----->  | Bounding Boxes + Class  |
+--------+---------+           +-------------------------+
         |
         v
+--------------------------+
| Depth Estimator (DAv2)  | --> Optional → Depth Map
+--------------------------+
         |
         v
+------------------------------+
| Fire Localization Strategy  |
| • Monocular Projection      |
| • Depth Estimation (DAv2)   |
+------------------------------+
         |
         v
+-----------------------------+
| Mission Planner (ROS 2)     |
| - Target selection          |
| - PX4 setpoint publishing   |
+-----------------------------+
         |
         v
+-----------------------------+
| RTSP Stream & GUI Feedback  |
+-----------------------------+


## Process Workflow

The FireDrone-X system follows this modular pipeline:

1. **Camera Input**  
   - Captures RGB images from a downward-facing high-res camera.

2. **Object Detection**  
   - YOLOv8 detects fire (cones) and humans in each frame.
   - Detections are filtered by confidence and class ID.

3. **Fire Localization**  
   Two localization strategies are implemented:
   - **Monocular Projection**  
     - Transforms pixel coordinates to ground-plane (x, y) using camera intrinsics and pitch angle.
   - **Monocular Depth Estimation**  
     - Uses DepthAnythingV2 model to generate depth map.
     - Extracts true 3D coordinates (x, y, z) of fire targets.

4. **Mission Planning**  
   - Selects nearest unvisited fire.
   - Sends UAV toward the fire via position setpoints.
   - Switches to circling once reached untill it finds next target.

5. **Visualization & Control**  
   - RTSP stream and real-time plots update the GUI.
   - GUI shows detection feed, fire list, and drone odometry.
   - Uses ROS 2 messages to publish control commands to PX4.

## Models & Datasets

- **Fire & Human Detection**  
  - Model: [YOLOv8](https://github.com/ultralytics/ultralytics)  
  - Datasets: COCO, VisDrone, and custom-simulated cone/person layouts

- **Monocular Depth Estimation**  
  - Model: [Depth Anything V2](https://github.com/isl-org/DPT) by [Intel ISL](https://www.intel.com/content/www/us/en/research/research-area/isl.html)  
  - Paper: *Depth Anything: Unleashing the Power of Large-Scale Image-Text Data for Monocular Depth Estimation*  
    ([arXiv:2402.13242](https://arxiv.org/abs/2402.13242))  
  - License: [MIT License](https://github.com/isl-org/Depth-Anything/blob/main/LICENSE)

---

## Hardware Platform

- **VOXL2 Flight Deck** by [ModalAI](https://www.modalai.com/products/voxl-2)  
  - SoC: Qualcomm QRB5165  
  - Integrated PX4 Autopilot 
  - Camera: Hires (Cam Config 11)  
  - Communication: WiFi and UART over VOXL2 IO board  
  - ROS 2 support via `voxl-mpa-to-ros2` bridge ([GitHub](https://github.com/modalai/voxl-mpa-to-ros2))

---

## Acknowledgments

We thank the following projects and teams for enabling this work:

- **Intel ISL** for open-sourcing the Depth Anything V2 model  
- **Ultralytics** for maintaining YOLOv8  
- **ModalAI** for providing the VOXL2 hardware and PX4 integration tools  
- **PX4 Autopilot** and **ROS 2 Humble** ecosystem for open-source autonomy development


