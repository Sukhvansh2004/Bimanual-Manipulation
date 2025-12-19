# Perception Pipeline Architecture

This document explains the perception pipeline architecture used in the Dual-Arm Franka Manipulation system.

## Overview

The perception system follows a modular microservices architecture with the following components:

1. **Object Detection** (FastSAM) - Segments objects from RGB images
2. **Point Cloud Generation** - Creates 3D point clouds from depth images
3. **Grasp Prediction** (GraspGen) - Predicts 6-DOF grasps from point clouds

## Architecture Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Teleoperation │───▶│  ObjectDetector  │───▶│ GraspPredictor  │
│   (User/Auto)   │    │   (FastSAM)      │    │   (GraspGen)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Camera Images │    │   Point Clouds   │    │   6-DOF Grasps  │
│   (RGB + Depth) │    │   (Filtered)     │    │   (Scored)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Component Details

### 1. ObjectDetector (FastSAM Integration)

**Location**: `src/robosuite_franka_manipulation/perception/object_detector.py`

**Functionality**:
- Loads FastSAM model from `models/FastSAM-x.pt`
- Uses default text prompt "object" (configurable)
- Runs FastSAM inference on RGB images
- Extracts segmentation masks
- Converts masks to 3D point clouds using depth images
- Calls local GraspGen predictor for grasp generation

**Key Methods**:
- `detect_objects()` - Main detection pipeline
- `_mask_to_point_cloud()` - Depth-based point cloud generation
- `_predict_grasps()` - Local grasp prediction using GraspGen

### 2. GraspPredictor (GraspGen Integration)

**Location**: `src/robosuite_franka_manipulation/perception/grasp_predictor.py`

**Functionality**:
- Loads GraspGen models from `deps/GraspGenModels/`
- Processes point clouds for grasp prediction
- Filters grasps for collision avoidance
- Returns scored 6-DOF grasp poses

### 3. Microservices

#### Prompt Service (Port 5557)
- **Protocol**: TCP socket
- **Commands**:
  - `SET: <prompt>` - Set detection prompt (e.g., "SET: red cube")
  - `GET` - Retrieve current prompt
- **Fallback**: Returns "object" if service unavailable

#### Grasp Prediction Service (Port 5558)
- **Protocol**: ZMQ REQ/REP
- **Input**: JSON with point cloud, object pose, grasp parameters
- **Output**: JSON with grasp poses and confidence scores

## Usage Examples

### Basic Pipeline Usage
```python
from robosuite_franka_manipulation.perception import ObjectDetector

# Setup
detector = ObjectDetector(env)

# Detect objects
objects = detector.detect_objects()

# Each object contains: pose, point_cloud, mask, grasps
for obj in objects:
    print(f"Found object with {len(obj['grasps'])} grasp predictions")
```

## Configuration

### FastSAM Parameters
- Model: `FastSAM-x.pt`
- Confidence threshold: 0.4
- IoU threshold: 0.9
- Image size: 1024x1024

### GraspGen Parameters
- Number of grasps: 10 (default)
- Grasp threshold: Model-dependent
- Collision checking: Enabled
- Gripper type: Franka Panda

### Camera Parameters
- Default intrinsics: fx=615, fy=615, cx=310, cy=236
- Depth scale: Automatic from sensor

## Troubleshooting

1. **No objects detected**: Check that FastSAM model loads correctly
2. **Empty point clouds**: Verify depth image availability in environment
3. **No grasps predicted**: Check GraspGen model and dependencies
4. **Import errors**: Ensure all dependencies are installed

## Future Improvements

1. **Real-time processing**: Optimize for lower latency
2. **Multi-object handling**: Better segmentation for complex scenes
3. **Grasp refinement**: Post-processing for grasp quality
4. **Sensor fusion**: Combine multiple camera views