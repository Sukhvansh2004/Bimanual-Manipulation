# Distributed Perception Services

Services contains a distributed microservices architecture for robotic perception and grasp planning. The system is designed for the Dual-Arm Franka manipulation setup with continuous camera processing, real-time visualization, and orchestrated grasp prediction.

## Architecture Overview

The system consists of three main components:

1. **Enhanced Service Manager** (`service_manager.py`) - Central orchestrator
2. **FastSAM Service** (`fastsam_service.py`) - Object detection and segmentation
3. **Grasp Service** (`grasp_service.py`) - Grasp prediction from point clouds

### Service Communication

- **Service Manager**: `tcp://localhost:5555`
- **FastSAM Service**: `tcp://localhost:5551`
- **Grasp Service**: `tcp://localhost:5552`

All communication uses ZeroMQ (ZMQ) with JSON message serialization.

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install zmq opencv-python numpy torch torchvision

# Install FastSAM and GraspGen dependencies
cd deps/GraspGen
pip install -r requirements.txt
python setup.py install
```

### 2. Start Services

```python
python scripts/demo_services.py
```

Or start services individually:

```python
from services.service_manager import EnhancedServiceManager
from services.fastsam_service import FastSAMService
from services.grasp_service import GraspService

# Start manager
manager = EnhancedServiceManager()
manager.start()

# Start FastSAM service
fastsam = FastSAMService(
    model_path='models/FastSAM-x.pt'
)
fastsam.start()

# Start Grasp service
grasp = GraspService(
    model_path='deps/GraspGenModels/checkpoints'
)
grasp.start()
```

### 3. Use the Services

```python
from services.service_manager import ServiceClient

client = ServiceClient()

# Set detection prompt
client.set_detection_prompt("red cup")

# Get complete perception pipeline
result = client.get_perception_pipeline(num_grasps=5)
print(f"Found {result['total_detections']} objects with {result['total_grasps']} grasps")
```

## Service Manager API

The `ServiceClient` provides a simple interface to all services:

### Service Registration & Discovery

```python
# Register a service
client.register("my_service", "localhost", 5556, "custom", "My custom service")

# Look up a service
service_info = client.lookup("fastsam_service")

# Get all service status
status = client.get_service_status()
```

### Perception Pipeline

```python
# Get complete pipeline results (detections + grasps)
result = client.get_perception_pipeline(num_grasps=5)

# Result structure:
{
    "status": "success",
    "pipeline_results": [
        {
            "detection": {...},  # Detection info
            "grasps": [...],     # List of grasp poses
            "pointcloud_size": 1024,
            "num_grasps": 5
        }
    ],
    "total_detections": 2,
    "total_grasps": 10
}
```

### Direct Service Calls

```python
# Call FastSAM service directly
response = client.call("fastsam_service", {
    "command": "GET_DETECTIONS"
})

# Call Grasp service directly
response = client.call("grasp_service", {
    "command": "PREDICT_GRASPS",
    "pointcloud": [...],
    "object_pose": {"pos": [x,y,z], "quat": [w,x,y,z]},
    "num_grasps": 5
})
```

## FastSAM Service

Provides continuous object detection and segmentation from camera feeds.

### Features

- **Continuous Processing**: Processes camera frames in real-time
- **OpenCV Visualization**: Shows detections with bounding boxes and masks
- **Point Cloud Extraction**: Converts depth images to 3D point clouds
- **Dynamic Prompts**: Update detection prompts without restarting
- **Multi-threading**: Separate threads for detection and visualization

### Configuration

```python
fastsam = FastSAMService(
    host='localhost',           # Service host
    port=5551,                  # Service port
    manager_host='localhost',   # Service manager host
    manager_port=5555,          # Service manager port
    model_path='models/FastSAM-x.pt',  # Model path
    camera_id=0,                # Camera ID (for OpenCV)
    conf_threshold=0.3,         # Confidence threshold
    iou_threshold=0.9,          # IoU threshold
    visualization=True,         # Enable OpenCV visualization
    max_fps=10                  # Maximum processing FPS
)
```

### API Commands

- `GET_STATUS`: Get service status and configuration
- `SET_PROMPT`: Update detection prompt
- `GET_DETECTIONS`: Get current detections
- `GET_POINTCLOUDS`: Get point clouds for detections
- `START_PROCESSING`: Start camera processing
- `STOP_PROCESSING`: Stop camera processing

## Grasp Service

Provides grasp prediction for segmented point clouds using GraspGen.

### Features

- **Batch Processing**: Process multiple point clouds efficiently
- **World Frame Transformation**: Convert grasps to world coordinates
- **Confidence Scoring**: Rank grasps by quality
- **Pose Validation**: Filter invalid grasp poses
- **Multi-threading**: Concurrent grasp prediction

### Configuration

```python
grasp = GraspService(
    host='localhost',           # Service host
    port=5552,                  # Service port
    manager_host='localhost',   # Service manager host
    manager_port=5555,          # Service manager port
    model_path='deps/GraspGenModels/checkpoints',  # Model directory
    batch_size=4,               # Batch size for prediction
    num_workers=2,              # Number of worker threads
    device='cuda'               # Device for inference
)
```

### API Commands

- `GET_STATUS`: Get service status and configuration
- `PREDICT_GRASPS`: Predict grasps for point cloud
- `BATCH_PREDICT`: Predict grasps for multiple point clouds

### Grasp Prediction Request

```python
{
    "command": "PREDICT_GRASPS",
    "pointcloud": [[x,y,z], ...],  # List of 3D points
    "object_pose": {
        "pos": [x, y, z],        # Object position
        "quat": [w, x, y, z]     # Object orientation (quaternion)
    },
    "num_grasps": 5              # Number of grasps to generate
}
```

### Grasp Response

```python
{
    "status": "success",
    "grasps": [
        {
            "pose": {
                "pos": [x, y, z],
                "quat": [w, x, y, z]
            },
            "score": 0.95,
            "width": 0.08
        }
    ]
}
```

## Integration with Robosuite

The services are designed to work with Robosuite environments:

```python
import robosuite as suite

# Create environment with camera
env = suite.make(
    "DualArmPanda",
    robots=["Panda", "Panda"],
    camera_names=["frontview"],
    camera_heights=480,
    camera_widths=640,
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True
)

# Services will automatically connect to camera feeds
# Use ServiceClient to get perception results
```

## Testing

Run the comprehensive test suite:

```bash
# Run all service tests
python -m pytest tests/test_services.py -v

# Run specific service tests
python -m pytest tests/test_services.py::TestFastSAMService -v
python -m pytest tests/test_services.py::TestGraspService -v
python -m pytest tests/test_services.py::TestServiceManager -v
```

## Performance Considerations

- **FastSAM Service**: Processes at ~10 FPS with visualization, ~30 FPS without
- **Grasp Service**: ~100ms per grasp prediction on GPU
- **Service Manager**: Sub-millisecond latency for orchestration
- **Memory Usage**: ~2GB for FastSAM, ~1GB for Grasp service

## Troubleshooting

### Common Issues

1. **Service Registration Fails**
   - Check that Service Manager is running on port 5555
   - Verify network connectivity

2. **Camera Not Found**
   - Ensure camera is properly connected
   - Check camera permissions
   - Use `v4l2-ctl --list-devices` to list cameras

3. **Model Loading Errors**
   - Verify model paths are correct
   - Check GPU memory availability
   - Ensure all dependencies are installed

4. **ZMQ Connection Errors**
   - Check firewall settings
   - Verify ports are not in use
   - Use `netstat -tlnp | grep :5555` to check port status

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Monitoring

The Service Manager automatically monitors service health:

```python
status = client.get_service_status()
for service_name, info in status['services'].items():
    print(f"{service_name}: {info['status']}")
```

## Development

### Adding New Services

1. Create service class inheriting from base service pattern
2. Implement required API commands
3. Register with Service Manager
4. Add to ServiceClient methods

### Extending the Pipeline

The orchestrated pipeline can be extended by:

1. Adding new services for additional processing steps
2. Modifying the `_handle_perception_pipeline` method
3. Updating the ServiceClient interface

## License

This code is part of the Dual-Arm-Franka-Control-MTP project.