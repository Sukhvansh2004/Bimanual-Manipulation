import numpy as np
from typing import List, Dict, TypedDict
import sys
import cv2
import logging
from PIL import Image

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.transforms import StampedPose, Transform
from robosuite_franka_manipulation.utils.pointcloud import (
    estimate_pose_from_point_cloud,
    transform_point_cloud,
    deproject_depth_to_point_cloud,
    parse_camera_intrinsics,
    ensure_depth_2d,
)

from ultralytics.models.fastsam import FastSAM
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


class Detection(TypedDict):
    """Type definition for detected object."""
    name: str
    pose: StampedPose
    pcd: np.ndarray
    mask: np.ndarray
    id: int

class ObjectDetector:
    """
    Class to detect objects in the scene using FastSAM segmentation.
    Pipeline: FastSAM detection -> Point cloud from depth -> Local grasp prediction
    """
    def __init__(
        self,
        camera_intrinsics: np.ndarray = constants.DEFAULT_CAMERA_INTRINSICS,
        camera_width: int = constants.DEFAULT_CAMERA_WIDTH,
        camera_height: int = constants.DEFAULT_CAMERA_HEIGHT,
        prompt: str = constants.DEFAULT_DETECTION_PROMPT,
        debug: bool = True,
        visualize: bool = False
    ) -> None:
        self.camera_intrinsics = camera_intrinsics
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.prompt = prompt
        self.debug = debug
        self.visualize = visualize
        
        # Track visualization window names for cleanup when prompt changes
        self._current_window_names: List[str] = []

        # Add GraspGen to path for FastSAM
        GRASPGEN_PATH = constants.get_project_root() / constants.GRASPGEN_DEPS_DIR
        if str(GRASPGEN_PATH) not in sys.path:
            sys.path.append(str(GRASPGEN_PATH))

        # Elevate logger level if debug enabled
        if self.debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        # Load FastSAM model from models folder
        model_path = constants.get_project_root() / constants.FASTSAM_MODEL_PATH
        self.model = FastSAM(str(model_path))  # Load model

    def set_prompt(self, prompt: str) -> None:
        """Set the detection prompt and close old visualization windows."""
        if prompt != self.prompt:
            # Close old visualization windows before changing prompt
            self._close_visualization_windows()
        self.prompt = prompt
    
    def _close_visualization_windows(self) -> None:
        """Close all tracked visualization windows."""
        for window_name in self._current_window_names:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass  # Window may already be closed
        self._current_window_names = []

    def _get_intrinsic_params(self) -> tuple:
        """Return (fx, fy, cx, cy) from stored intrinsics. Raises error if intrinsics invalid."""
        fx, fy, cx, cy = parse_camera_intrinsics(self.camera_intrinsics)
        if self.debug:
            logger.debug(f"Using intrinsics fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        return fx, fy, cx, cy

    def _ensure_depth_2d(self, depth_image: np.ndarray) -> np.ndarray:
        """Convert depth array to 2D. Raises error if malformed."""
        return ensure_depth_2d(depth_image, height=self.camera_height, width=self.camera_width)

    def detect_objects(self, rgb_image: np.ndarray, depth_image: np.ndarray,
                      camera_pos: np.ndarray, camera_rot: R) -> List[Detection]:
        """
        Detect objects in the scene using FastSAM on camera images.

        Args:
            rgb_image: RGB image from camera
            depth_image: Depth image from camera (required for point clouds)
            camera_pos: Camera position in world frame
            camera_rot: Camera rotation (scipy Rotation object)

        Returns:
            List of detected objects with poses, point clouds, and masks
        """
        objects: List[Detection] = []

        if not self.model:
            logger.error("FastSAM model not available")
            return objects

        # Normalize depth to 2D
        depth_image = self._ensure_depth_2d(depth_image)

        # Run FastSAM with text prompt
        if self.debug:
            logger.debug(f"Running FastSAM with prompt: '{self.prompt}'")
        
        # Convert numpy array to PIL Image for FastSAM
        pil_image = Image.fromarray(rgb_image)
        results = self.model(
            pil_image,
            device=constants.DEVICE,
            retina_masks=True,
            imgsz=1024,
            conf=constants.FASTSAM_CONF_THRESHOLD,
            iou=constants.FASTSAM_IOU_THRESHOLD,
            texts=self.prompt,
            verbose=False,
        )

        if not results or len(results) == 0:
            logger.debug("No FastSAM results")
            return objects

        result = results[0]
        if result.masks is None or len(result.masks) == 0:
            logger.debug("No masks detected")
            return objects

        masks = result.masks.data.cpu().numpy()
        
        if self.debug:
            logger.debug(f"FastSAM detected {len(masks)} objects with prompt '{self.prompt}'")
        
        if self.visualize:
            self._visualize_fastsam_results(result, rgb_image)

        # Process each detected object
        for i, mask in enumerate(masks):
            # Create point cloud from mask and depth in camera frame
            pcd_camera_frame = self._mask_to_point_cloud(mask, depth_image)
            
            if pcd_camera_frame.size == 0:
                continue

            # Create transform from camera frame to world frame
            camera_to_world = Transform(
                translation=camera_pos,
                rotation=camera_rot.as_quat(),
                parent_frame="world",
                child_frame="camera"
            )
            
            # Transform point cloud to world frame using matrix for efficiency
            transform_matrix = camera_to_world.to_matrix()
            pcd_world_frame = transform_point_cloud(pcd_camera_frame, transform_matrix)
            
            # Estimate pose from point cloud in world frame
            pose = self._estimate_pose_from_point_cloud(pcd_world_frame)

            objects.append({
                'name': f'object_{i}',
                'pose': pose,
                'pcd': pcd_world_frame,
                'mask': mask,
                'id': i
            })

        return objects

    def _mask_to_point_cloud(self, mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Convert mask to point cloud using depth and camera intrinsics."""
        # Ensure depth is 2D; reshape flat buffers when dimensions are known
        depth = np.asarray(depth)
        if depth.ndim == 1:
            expected = self.camera_height * self.camera_width
            if depth.size != expected:
                raise ValueError(f"Flat depth buffer size {depth.size} != expected {expected}")
            depth = depth.reshape(self.camera_height, self.camera_width)

        depth = self._ensure_depth_2d(depth)
        
        if self.debug:
            logger.debug(f"mask_to_point_cloud depth shape={depth.shape}, ndim={depth.ndim}")

        fx, fy, cx, cy = self._get_intrinsic_params()

        # Use utility function for deprojection
        # Note: image_flipped=False because the flipped RGB/depth images from
        # fastsam_service are now in standard OpenCV convention (origin top-left),
        # which matches what the robosuite extrinsic matrix (with axis correction)
        # expects. The pinhole deprojection formula uses v directly without
        # additional conversion.
        return deproject_depth_to_point_cloud(
            depth, mask, fx, fy, cx, cy, 
            max_depth=constants.MAX_POINTCLOUD_RANGE,
            mujoco_convention=False,
            image_flipped=False,
        )

    def _estimate_pose_from_point_cloud(self, pcd: np.ndarray, frame_id: str = "world") -> StampedPose:
        """Estimate object pose from point cloud using PCA for orientation."""
        pos, quat = estimate_pose_from_point_cloud(pcd)
        return StampedPose(position=pos, quaternion=quat, frame_id=frame_id)

    def _visualize_fastsam_results(self, result, rgb_image: np.ndarray) -> None:
        """
        Visualize FastSAM detection results using Ultralytics built-in plot function.
        Shows both raw camera image and annotated segmentation results.

        Args:
            result: FastSAM result object from model inference
            rgb_image: Original RGB image from camera
        """
        if not self.visualize or result is None:
            return
        
        # Define window names
        camera_window = f"Camera View: {self.prompt}"
        segmentation_window = f"FastSAM Segmentation: {self.prompt}"
        
        # Track current window names for cleanup
        new_window_names = [camera_window, segmentation_window]
        if self._current_window_names != new_window_names:
            self._close_visualization_windows()
            self._current_window_names = new_window_names
        
        # Show raw camera image
        cv2.imshow(camera_window, rgb_image)
        
        # Show annotated segmentation results
        annotated_image = result.plot()
        cv2.imshow(segmentation_window, annotated_image)
        
        cv2.waitKey(1)  # Refresh windows
