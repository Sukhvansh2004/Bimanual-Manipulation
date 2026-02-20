#!/usr/bin/env python3
"""
System Manager for Dual-Arm Manipulation.
Provides a unified interface for managing environment, controllers, and services.
"""

import threading
import time
import logging
import os
import signal
import sys
from typing import Any, Optional, List
import numpy as np
import meshcat
from scipy.spatial.transform import Rotation as R

from robosuite.wrappers import VisualizationWrapper
import robosuite_franka_manipulation.constants as constants
from robosuite_franka_manipulation.environments.dual_arm import make_dual_arm_env, DualArmEnvironment
from robosuite_franka_manipulation.teleop.teleop import KeyboardTeleop
from robosuite_franka_manipulation.controllers.arm import RobosuiteArmController
from robosuite_franka_manipulation.controllers.gripper import RobosuiteGripperController
from robosuite_franka_manipulation.services.fastsam_service import FastSAMService
from robosuite_franka_manipulation.services.grasp_service import GraspService
from robosuite_franka_manipulation.services.move_to_pose_service import MoveToPoseService
from robosuite_franka_manipulation.services.services_core import ServiceManager, ServiceClient
from robosuite_franka_manipulation.transforms.stamped_pose import StampedPose, Transform
from robosuite_franka_manipulation.transforms.transform_manager import TransformManager
from robosuite_franka_manipulation.transforms.meshcat_tf_visualizer import (
    MeshcatTFVisualizerConfig,
    stream_transforms_to_meshcat,
)
from robosuite_franka_manipulation.utils.env_utils import env_step, set_robot_config
from robosuite_franka_manipulation.planning.utils.trajectory_utils import (
    load_trajectory, 
    WaypointFeasibilityResult
)
from robosuite_franka_manipulation.planning.utils.visualization_utils import visualize_trajectory

logger = logging.getLogger(__name__)

class DualArmManipulationSystem:
    """
    Complete system for dual-arm manipulation with services.
    """
    
    def __init__(
            self, 
            visualize: bool = False, 
            debug: bool = False, 
            detection_prompt: str = "object", 
            meshcat_tf: bool = False,
            custom_objects: Optional[Any] = None,
            ycb_objects: Optional[Any] = None,
            use_fastsam: bool = True,
            use_grasp_service: bool = True,
            has_renderer: bool = True,
            ):
        
        self.custom_objects = custom_objects
        self.ycb_objects = ycb_objects
        self.visualize = visualize
        self.debug = debug
        self.detection_prompt = detection_prompt
        self.meshcat_tf = meshcat_tf
        self.use_fastsam = use_fastsam
        self.use_grasp_service = use_grasp_service
        self.has_renderer = has_renderer
        # Use RLock to prevent deadlocks (TransformManager re-acquires it)
        self.env_lock = threading.RLock()

        if self.debug:
            logger.setLevel(logging.DEBUG)

        self.env: DualArmEnvironment = None
        self.tf_manager: Optional[TransformManager] = None
        self.teleop: Optional[KeyboardTeleop] = None
        self.teleop_thread: Optional[threading.Thread] = None
        self._meshcat_thread: Optional[threading.Thread] = None
        
        # Arm and gripper controllers
        self.left_arm: Optional[RobosuiteArmController] = None
        self.right_arm: Optional[RobosuiteArmController] = None
        self.left_gripper: Optional[RobosuiteGripperController] = None
        self.right_gripper: Optional[RobosuiteGripperController] = None
        
        # Services
        self.service_manager: Optional[ServiceManager] = None
        self.fastsam_service: Optional[FastSAMService] = None
        self.grasp_service: Optional[GraspService] = None
        self.move_service: Optional[MoveToPoseService] = None
        self.service_client: Optional[ServiceClient] = None
            
        self.running = False

        # Shutdown coordination
        self._stop_lock = threading.Lock()
        self._shutdown_requested = threading.Event()

    def _is_quit_key(self, key: Any) -> bool:
        """Best-effort check for 'q' across different key callback signatures."""
        try:
            if isinstance(key, str):
                return key.lower() == 'q'
            if isinstance(key, int):
                try:
                    return chr(key).lower() == 'q'
                except Exception:
                    return False
            if hasattr(key, 'char') and isinstance(key.char, str):
                return key.char.lower() == 'q'
        except Exception:
            return False
        return False

    def request_shutdown(self, reason: str = "", force_after_s: float = 1.5) -> None:
        """Request shutdown from callbacks/signals without blocking the caller."""
        if self._shutdown_requested.is_set():
            return
        self._shutdown_requested.set()
        if reason:
            logger.info("Shutdown requested: %s", reason)

        def _shutdown_worker() -> None:
            try:
                self.stop()
            finally:
                if force_after_s is not None and force_after_s > 0:
                    time.sleep(force_after_s)
                os._exit(0)

        threading.Thread(target=_shutdown_worker, daemon=True).start()
        
    def start(self, start_teleop_thread: bool = False):
        """Start the complete system. Optionally start teleop in a separate thread."""
        if self.running:
            logger.warning("System already running")
            return
            
        logger.info("="*70)
        logger.info("Starting Dual-Arm Manipulation System")
        logger.info("="*70)
        
        # Create environment
        logger.info("\n1. Creating dual-arm environment...")
        self.env = make_dual_arm_env(
            custom_objects=self.custom_objects, 
            ycb_objects=self.ycb_objects, 
            has_renderer=self.has_renderer
        )
        if self.has_renderer:
            logger.debug("Adding VisualizationWrapper to environment...")
            self.env = VisualizationWrapper(self.env)
        logger.info("   ✓ Environment created")

        # Shared TF manager for the whole system (services + controllers + visualization)
        logger.info("\n1a. Creating shared TransformManager...")
        self.tf_manager = TransformManager(env=self.env, auto_discover=True, env_lock=self.env_lock)
        logger.info("   ✓ TransformManager created")

        # Optional MeshCat TF visualizer streaming the SAME tf_manager/env
        if self.meshcat_tf:
            logger.info("   Starting MeshCat TF visualizer (meshcat-server must be running)...")

            def _run_meshcat_tf() -> None:
                assert self.env is not None
                assert self.tf_manager is not None
                cfg = MeshcatTFVisualizerConfig(fps=20.0, clear=True, labels=True, visualize_scene=True)
                stream_transforms_to_meshcat(
                    env=self.env,
                    env_lock=self.env_lock,
                    tf_manager=self.tf_manager,
                    config=cfg,
                    stop_event=self._shutdown_requested,
                )

            self._meshcat_thread = threading.Thread(target=_run_meshcat_tf, daemon=True)
            self._meshcat_thread.start()
        
        # Load arm and gripper controllers
        logger.info("\n1b. Loading arm and gripper controllers...")
        self.left_arm = RobosuiteArmController(
            self.env, arm_id="right", robot_id=0, env_lock=self.env_lock, 
            transform_manager=self.tf_manager, visualize=self.visualize
        )
        self.right_arm = RobosuiteArmController(
            self.env, arm_id="right", robot_id=1, env_lock=self.env_lock, 
            transform_manager=self.tf_manager, visualize=self.visualize
        )
        self.left_gripper = RobosuiteGripperController(
            self.env, arm_id="right", robot_id=0, env_lock=self.env_lock
        )
        self.right_gripper = RobosuiteGripperController(
            self.env, arm_id="right", robot_id=1, env_lock=self.env_lock
        )
        logger.info("   ✓ Controllers loaded")
        
        # Start services
        logger.info("\n2. Starting services...")
        self._start_services()
        
        # Prepare teleoperation
        logger.info("\n3. Preparing teleoperation...")
        self.teleop = KeyboardTeleop(self.env, env_lock=self.env_lock)
        if start_teleop_thread:
            self.run_teleop(blocking=False)
            logger.info("   ✓ Teleoperation started (thread)")
        else:
            logger.info("   Teleoperation ready. Run system.run_teleop() to start.")
        
        # Settle steps
        logger.info("\n4. Settling environment...")
        for _ in range(10):
            with self.env_lock:
                env_step(self.env)
        
        # Register dynamic TF frames for every object in the scene
        self._register_object_frames()

        self.running = True
        
        logger.info("\n" + "="*70)
        logger.info("System Ready!")
        logger.info("="*70)

    def _register_object_frames(self):
        """Register dynamic TF frames for every object in the scene."""
        if self.tf_manager is None or self.env is None:
            return

        for name in self.env.get_object_names():
            frame_id = name

            def _make_provider(obj_name, fid):
                def provider() -> Transform:
                    with self.env_lock:
                        pose_4x4 = self.env.get_object_pose(obj_name)
                    if pose_4x4 is None:
                        return Transform.identity("world", fid)
                    pos = pose_4x4[:3, 3]
                    quat_xyzw = R.from_matrix(pose_4x4[:3, :3]).as_quat()
                    return Transform(
                        translation=pos,
                        rotation=quat_xyzw,
                        parent_frame="world",
                        child_frame=fid,
                    )
                return provider

            self.tf_manager.add_dynamic_transform(
                frame_id=frame_id,
                parent_frame="world",
                provider=_make_provider(name, frame_id),
            )
            logger.info(f"   Registered dynamic TF frame '{frame_id}'")
        
    def _start_services(self):
        """Start all services."""
        try:
            # Start Service Manager
            logger.info("   Starting Service Manager...")
            self.service_manager = ServiceManager()
            self.service_manager.start()
            time.sleep(0.5)
            logger.info("   ✓ Service Manager started")
            
            # Initialize client
            self.service_client = ServiceClient()
            
            # Start FastSAM service
            if self.use_fastsam:
                logger.info("   Starting FastSAM service...")
                assert self.env is not None
                self.fastsam_service = FastSAMService(
                    visualize=self.visualize,
                    debug=False,
                    tf_manager=self.tf_manager,
                )
                self.fastsam_service.start()
                time.sleep(0.5)
                
                # Set initial prompt
                self.service_client.call('fastsam_service', {
                    'command': 'SET_PROMPT',
                    'prompt': self.detection_prompt
                })
                logger.info(f"   ✓ FastSAM service started (prompt: '{self.detection_prompt}')")
            else:
                logger.info("   FastSAM service disabled by flag")
            
            # Start Grasp service
            if self.use_grasp_service:
                logger.info("   Starting Grasp service...")
                self.grasp_service = GraspService(
                    tf_manager=self.tf_manager,
                    debug=self.debug,
                    visualize=self.visualize,
                )
                self.grasp_service.start()
                time.sleep(0.5)
                logger.info("   ✓ Grasp service started")
            else:
                logger.info("   Grasp service disabled by flag")
            
            # Start Move To Pose service
            logger.info("   Starting Move To Pose service...")
            self.move_service = MoveToPoseService()
            # Register controllers
            if self.left_arm:
                self.move_service.register_controller(0, "right", self.left_arm)
            if self.right_arm:
                self.move_service.register_controller(1, "right", self.right_arm)
            
            self.move_service.start()
            time.sleep(0.5)
            logger.info("   ✓ Move To Pose service started")
            
        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            raise
    
    def _run_teleop(self):
        """Run teleoperation loop."""
        try:
            assert self.env is not None, "Environment not initialized"
            assert self.teleop is not None, "Teleop not initialized"
            
            if hasattr(self.env, "viewer") and self.env.viewer is not None:
                self.env.viewer.add_keypress_callback(self.teleop.device.on_press)  # type: ignore[union-attr]
                
                def _on_viewer_keypress(key: Any) -> None:
                    if self._is_quit_key(key):
                        self.request_shutdown(reason="q pressed")

                self.env.viewer.add_keypress_callback(_on_viewer_keypress)
            
            self.teleop.run()
            
        except Exception as e:
            logger.error(f"Teleoperation error: {e}", exc_info=True)
        finally:
            self._stop_teleop()

    def run_teleop(self, blocking: bool = True):
        """Public method to run teleoperation."""
        if blocking:
            self._run_teleop()
        else:
            self.teleop_thread = threading.Thread(target=self._run_teleop, daemon=True)
            self.teleop_thread.start()
    
    def _stop_teleop(self):
        """Stop teleoperation."""
        if self.teleop is None:
            return
        try:
            self.teleop.exit_pressed = True
            self.teleop.running = False
        except Exception as e:
            logger.debug(f"Error during teleop cleanup: {e}")

    def stop(self):
        """Stop the system."""
        with self._stop_lock:
            if not self.running and not self._shutdown_requested.is_set():
                return
            
        logger.info("\nStopping system...")
        self._stop_teleop()
        
        if self.move_service:
            self.move_service.stop()
        if self.fastsam_service:
            self.fastsam_service.stop()
        if self.grasp_service:
            self.grasp_service.stop()
        if self.service_manager:
            self.service_manager.stop()

        if self.service_client and hasattr(self.service_client, "close"):
            try:
                self.service_client.close()
            except Exception:
                pass

        if self.env is not None:
            try:
                viewer = getattr(self.env, "viewer", None)
                if viewer is not None:
                    if hasattr(viewer, "close"):
                        viewer.close()
                    if hasattr(viewer, "finish"):
                        viewer.finish()
            except Exception:
                pass
            try:
                if hasattr(self.env, "close"):
                    self.env.close()
            except Exception:
                pass
        
        self.running = False
        logger.info("System stopped")

# Helper functions originally in scripts
def detect(system: DualArmManipulationSystem):
    """Quick helper to detect objects."""
    if not system.use_fastsam:
        logger.info("FastSAM service is disabled.")
        return []
    assert system.service_client is not None, "Service client not initialized"
    response = system.service_client.call('fastsam_service', {'command': 'GET_LATEST_DETECTION'})
    if response.get('status') == 'success':
        detections = response.get('detections', [])
        logger.info(f"\nDetected {len(detections)} object(s):")
        for i, det in enumerate(detections):
            pc_size = len(det.get('pcd', []))
            pose = det.get('pose', {})
            logger.info(f"  Object {i+1}: {pc_size} points, pos={pose.get('pos', 'N/A')}")
        return detections
    else:
        logger.info(f"Detection failed: {response.get('message')}")
        return []

def get_grasps(system: DualArmManipulationSystem, detection, gripper_type, num_grasps=constants.DEFAULT_NUM_GRASPS, collision_check: Optional[str] = None):
    """Quick helper to generate grasps for a detection."""
    pcd = detection.get('pcd', [])
    pose = detection.get('pose', {})
    
    if not pcd or not pose:
        logger.info("Invalid detection - missing pcd or pose")
        return []
    
    assert system.service_client is not None, "Service client not initialized"
    response = system.service_client.call('grasp_service', {
        'command': 'GET_GRASPS',
        'pcd': pcd,
        'object_pose': pose,
        'gripper_type': gripper_type,
        'num_grasps': num_grasps,
        'collision_check': collision_check
    })
    
    if response.get('status') == 'success':
        grasps = response.get('grasps', [])
        logger.info(f"\nGenerated {len(grasps)} grasp(s):")
        for i, grasp in enumerate(grasps[:3]):
            logger.info(f"  Grasp {i+1}: pos={grasp.get('pos')}, score={grasp.get('score', 'N/A'):.3f}")
        return grasps
    else:
        logger.info(f"Grasp generation failed: {response.get('message')}")
        return []

def get_grasps_mesh(system: DualArmManipulationSystem, pose: StampedPose, object_name: str, num_grasps=constants.DEFAULT_NUM_GRASPS, gripper_type='suction', collision_check: Optional[str] = None, visualize_gripper_mesh=True):
    """Quick helper to generate grasps given an object pose."""
    assert system.service_client is not None, "Service client not initialized"
    response = system.service_client.call('grasp_service', {
        'command': 'GET_GRASPS',
        'object_pose': pose.model_dump(),
        'object_name': object_name,
        'num_grasps': num_grasps,
        'gripper_type': [gripper_type],
        'collision_check': collision_check,
        'visualize_gripper_mesh': visualize_gripper_mesh
    })
    
    if response.get('status') == 'success':
        grasps = response.get('grasps', [])
        logger.info(f"\nGenerated {len(grasps)} grasp(s):")
        for i, grasp in enumerate(grasps[:3]):
            logger.info(f"  Grasp {i+1}: pos={grasp.get('pos')}, score={grasp.get('score', 'N/A'):.3f}")
        return grasps
    else:
        logger.info(f"Grasp generation failed: {response.get('message')}")
        return []

def get_eef(system: DualArmManipulationSystem):
    """Quick helper to get end-effector coordinates."""
    if system.tf_manager is None:
        logger.info("TransformManager not initialized")
        return None
    try:
        eef_data = {}
        if system.env is None:
            logger.info("Environment not initialized")
            return None
        for robot_idx, robot in enumerate(system.env.robots):
            robot_name = f"robot{robot_idx}"
            try:
                arm_ids = list(robot.robot_model.eef_name.keys())
            except Exception:
                arm_ids = ["right"]
            for arm_id in arm_ids:
                frame_name = f"{robot_name}_{arm_id}_eef"
                if system.tf_manager.frame_exists(frame_name):
                    pose = system.tf_manager.get_frame_pose(frame_name, "world")
                    eef_data[frame_name] = {
                        "position": {"x": pose.position[0], "y": pose.position[1], "z": pose.position[2]},
                        "quaternion": {"x": pose.quaternion[0], "y": pose.quaternion[1], "z": pose.quaternion[2], "w": pose.quaternion[3]},
                        "frame_id": pose.frame_id
                    }
        logger.info("\nEnd-Effector Coordinates:")
        for frame_name, data in eef_data.items():
            pos = data.get('position', {})
            logger.info(f"  {frame_name}: x={pos.get('x', 0):.4f}, y={pos.get('y', 0):.4f}, z={pos.get('z', 0):.4f}")
        return eef_data
    except Exception as e:
        logger.info(f"Failed to get EEF: {e}")
        return None

def set_prompt(system: DualArmManipulationSystem, prompt: str):
    """Quick helper to change detection prompt."""
    if not system.use_fastsam:
        logger.info("FastSAM service is disabled.")
        return
    assert system.service_client is not None, "Service client not initialized"
    response = system.service_client.call('fastsam_service', {'command': 'SET_PROMPT', 'prompt': prompt})
    if response.get('status') == 'success':
        logger.info(f"Prompt changed to: '{prompt}'")
    else:
        logger.info(f"Failed to change prompt: {response.get('message')}")

def go_to_pose(system: DualArmManipulationSystem, pose: StampedPose, robot_id: int, arm_id: str, gripper_action: float = 0.0, num_steps: Optional[int] = None):
    """Quick helper to move an arm to a pose."""
    assert system.service_client is not None, "Service client not initialized"
    pose_dict = pose.model_dump()
    response = system.service_client.call('move_to_pose_service', {
        'command': 'MOVE_TO_POSE',
        'robot_id': robot_id,
        'arm': arm_id,
        'pose': pose_dict,
        'gripper_action': gripper_action,
        'num_steps': num_steps
    })
    if response.get('status') == 'success':
        logger.info(f"Move command sent to robot {robot_id}, arm {arm_id}")
    else:
        logger.info(f"Failed to send move command: {response.get('message')}")

def follow_eef_trajectory(system: DualArmManipulationSystem, robot_id: int, arm_id: str, trajectory: List[StampedPose], gripper_action: float = 0.0, num_steps: Optional[int] = None, smoothen: bool = False):
    """Quick helper to send a follow trajectory command."""
    assert system.service_client is not None, "Service client not initialized"
    traj_dicts = [p.model_dump() for p in trajectory]
    response = system.service_client.call('move_to_pose_service', {
        'command': 'FOLLOW_TRAJECTORY',
        'robot_id': robot_id,
        'arm': arm_id,
        'trajectory': traj_dicts,
        'gripper_action': gripper_action,
        'num_steps': num_steps,
        'smoothen': smoothen
    })
    if response.get('status') == 'success':
        logger.info(f"Follow trajectory command sent to robot {robot_id}, arm {arm_id}")
    else:
        logger.info(f"Failed to send follow trajectory command: {response.get('message')}")

def cancel_move(system: DualArmManipulationSystem, robot_id: int, arm_id: str):
    """Quick helper to cancel a move."""
    assert system.service_client is not None, "Service client not initialized"
    response = system.service_client.call('move_to_pose_service', {
        'command': 'CANCEL_MOVE',
        'robot_id': robot_id,
        'arm': arm_id
    })
    if response.get('status') == 'success':
        logger.info(f"Cancel command sent to robot {robot_id}, arm {arm_id}")
    else:
        logger.info(f"Failed to send cancel command: {response.get('message')}")

def get_arm_status(system: DualArmManipulationSystem, robot_id: int, arm_id: str):
    """Quick helper to get arm status."""
    assert system.service_client is not None, "Service client not initialized"
    response = system.service_client.call('move_to_pose_service', {
        'command': 'GET_ARM_STATUS',
        'robot_id': robot_id,
        'arm': arm_id
    })
    if response.get('status') == 'success':
        return response.get('status_code')
    else:
        logger.info(f"Failed to get arm status: {response.get('message')}")
        return None

def load_trajectory_file(filepath: str):
    """Load trajectory from file."""
    try:
        eef_traj, obj_traj, joint_configs, waypoint_results = load_trajectory(filepath)
        logger.info(f"Loaded trajectory from {filepath}: {len(eef_traj)} waypoints")
        feasibility = np.array([r.is_feasible for r in waypoint_results]) if waypoint_results else None
        return {
            "eef_traj": eef_traj, 
            "obj_traj": obj_traj,
            "joint_configs": joint_configs,
            "waypoint_results": waypoint_results,
            "feasibility":  feasibility
        }
    except Exception as e:
        logger.info(f"Failed to load trajectory: {e}")
        return None

def visualize_trajectory_data(system: DualArmManipulationSystem, eef_trajectory: List[StampedPose], waypoint_results: Optional[List[WaypointFeasibilityResult]] = None, name: str = "trajectory"):
    """Visualize trajectory data in MeshCat."""
    if system.tf_manager is None:
        logger.info("TransformManager not initialized")
        return
    vis = meshcat.Visualizer(zmq_url=str(constants.MESHCAT_ZMQ_URL))
    visualize_trajectory(vis, eef_trajectory, waypoint_results, name=name, tf_manager=system.tf_manager)
    logger.info(f"Trajectory visualization sent to MeshCat as '{name}'")
