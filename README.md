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



