#!/usr/bin/env python3
"""
Start Services and Interactive Teleoperation

This script:
1. Starts the dual-arm environment with teleoperation in a separate thread
2. Starts all services (ServiceManager, FastSAM, GraspGen, MoveToPose)
3. Provides an interactive Python shell for service queries

You can teleoperate the robot in the environment window using standard robosuite
keyboard controls, while simultaneously using the Python terminal to:
- Query detections: client.call('fastsam_service', {'command': 'GET_LATEST_DETECTION'})
- Generate grasps: client.call('grasp_service', {'command': 'GET_GRASPS', ...})
- Change prompts: client.call('fastsam_service', {'command': 'SET_PROMPT', 'prompt': 'cup'})

Usage:
    python scripts/manipulation/dual_arm_manipulation.py
    
    Then in the interactive shell:
    >>> # Get latest detections
    >>> response = client.call('fastsam_service', {'command': 'GET_LATEST_DETECTION'})
    >>> print(response)
    
    >>> # Change detection prompt
    >>> client.call('fastsam_service', {'command': 'SET_PROMPT', 'prompt': 'red cup'})
    
    >>> # Generate grasps (after detecting objects)
    >>> detections = response['detections']
    >>> if detections:
    ...     pc = detections[0]['pcd']
    ...     pose = detections[0]['pose']
    ...     grasps = client.call('grasp_service', {
    ...         'command': 'GET_GRASPS',
    ...         'pcd': pc,
    ...         'object_pose': pose,
    ...         'num_grasps': 5
    ...     })
    ...     print(grasps)
"""

import argparse
import threading
import time
import logging
import code
import atexit
import os
import signal
import sys
from typing import Any, Optional

from robosuite.wrappers import VisualizationWrapper
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.teleop import KeyboardTeleop
from robosuite_franka_manipulation.controllers import RobosuiteArmController, RobosuiteGripperController
from robosuite_franka_manipulation.services.fastsam_service import FastSAMService
from robosuite_franka_manipulation.services.grasp_service import GraspService
from robosuite_franka_manipulation.services.move_to_pose_service import MoveToPoseService
from robosuite_franka_manipulation.services.services_core import ServiceManager, ServiceClient
from robosuite_franka_manipulation.transforms import TransformManager, MeshcatTFVisualizerConfig, stream_transforms_to_meshcat
from robosuite_franka_manipulation.transforms.stamped_pose import StampedPose
from robosuite_franka_manipulation.utils.env_utils import create_ycb_object, env_step

# Configure logging
logging.basicConfig(
    level=constants.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,
)
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
            ):
        
        self.custom_objects = custom_objects
        self.ycb_objects = ycb_objects
        self.visualize = visualize
        self.debug = debug
        self.detection_prompt = detection_prompt
        self.meshcat_tf = meshcat_tf
        # Use RLock to prevent deadlocks (TransformManager re-acquires it)
        self.env_lock = threading.RLock()

        if self.debug:
            logger.setLevel(logging.DEBUG)

        self.env: Any = None
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
            # Common robosuite/glfw patterns: int keycode or single-char string
            if isinstance(key, str):
                return key.lower() == 'q'
            if isinstance(key, int):
                try:
                    return chr(key).lower() == 'q'
                except Exception:
                    return False
            # pynput-style key objects
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
                # If something is wedged (viewer/render), force an exit after a short grace period.
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
        self.env = make_dual_arm_env(custom_objects=self.custom_objects, ycb_objects=self.ycb_objects)
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
                cfg = MeshcatTFVisualizerConfig(fps=20.0, clear=True, labels=True)
                stream_transforms_to_meshcat(
                    env=self.env,
                    env_lock=self.env_lock,
                    tf_manager=self.tf_manager,
                    auto_discover=True,
                    config=cfg,
                    stop_event=self._shutdown_requested,
                )

            self._meshcat_thread = threading.Thread(target=_run_meshcat_tf, daemon=True)
            self._meshcat_thread.start()
        
        # Load arm and gripper controllers
        logger.info("\n1b. Loading arm and gripper controllers...")
        self.left_arm = RobosuiteArmController(
            self.env, arm_id="right", robot_id=0, env_lock=self.env_lock, transform_manager=self.tf_manager, visualize=self.visualize
        )
        self.right_arm = RobosuiteArmController(
            self.env, arm_id="right", robot_id=1, env_lock=self.env_lock, transform_manager=self.tf_manager, visualize=self.visualize
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
        
        self.running = True
        
        logger.info("\n" + "="*70)
        logger.info("System Ready!")
        logger.info("="*70)
        logger.info("\nTeleoperation Controls (in environment window):")
        logger.info("  Arrow keys, .-;, o-p, y-h, e-r - Move robot")
        logger.info("  spacebar                       - Toggle gripper")
        logger.info("  s                              - Switch active arm")
        logger.info("  q                              - Exit teleoperation")
        logger.info("\nAvailable Controllers (in interactive shell):")
        logger.info("  system.left_arm                - RobosuiteArmController for left arm")
        logger.info("  system.right_arm               - RobosuiteArmController for right arm")
        logger.info("  system.left_gripper            - RobosuiteGripperController for left gripper")
        logger.info("  system.right_gripper           - RobosuiteGripperController for right gripper")
        logger.info("\nInteractive Python Shell Commands:")
        logger.info("  client                         - ServiceClient instance")
        logger.info("  detect()                       - Quick detect objects")
        logger.info("  get_grasps(detection)          - Generate grasps for detection")
        logger.info("  get_eef()                      - Get end-effector coordinates")
        logger.info("  set_prompt(text)               - Change detection prompt")
        logger.info("="*70 + "\n")
        
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
            logger.info("   Starting FastSAM service...")
            assert self.env is not None
            self.fastsam_service = FastSAMService(
                env=self.env,
                visualize=self.visualize,
                debug=False,
                env_lock=self.env_lock,
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
            
            # Start Grasp service
            logger.info("   Starting Grasp service...")
            self.grasp_service = GraspService(
                env=self.env,
                env_lock=self.env_lock,
                tf_manager=self.tf_manager,
                debug=self.debug,
                visualize=self.visualize,
            )
            self.grasp_service.start()
            time.sleep(0.5)
            logger.info("   ✓ Grasp service started")
            
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
        """
        Run teleoperation loop.
        
        Args:
            blocking: If True, runs in current thread (for main thread usage).
                      If False, intended to be called from a daemon thread.
        """
        try:
            assert self.env is not None, "Environment not initialized"
            assert self.teleop is not None, "Teleop not initialized"
            
            # Add keypress callback for robosuite device
            self.env.viewer.add_keypress_callback(self.teleop.device.on_press)  # type: ignore[union-attr]
            
            # Handle quit in the viewer callback so 'q' works reliably when the
            # mujoco window has focus - triggers immediate shutdown of services
            def _on_viewer_keypress(key: Any) -> None:
                if self._is_quit_key(key):
                    self.request_shutdown(reason="q pressed")

            self.env.viewer.add_keypress_callback(_on_viewer_keypress)
            
            # Run teleoperation guarded by env_lock
            self.teleop.run()
            
        except Exception as e:
            logger.error(f"Teleoperation error: {e}")
        finally:
            # Ensure teleop resources are cleaned up
            self._stop_teleop()

    def run_teleop(self, blocking: bool = True):
        """
        Public method to run teleoperation.
        
        Args:
            blocking: If True (default), runs in current thread.
                      If False, starts in a daemon thread and returns immediately.
        """
        if blocking:
            self._run_teleop()
        else:
            self.teleop_thread = threading.Thread(target=self._run_teleop, daemon=True)
            self.teleop_thread.start()
    
    def _stop_teleop(self):
        """Stop teleoperation and clean up keyboard/pynput resources."""
        if self.teleop is None:
            return
            
        try:
            # Signal teleop to exit
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
        
        # Stop teleop and keyboard/pynput resources
        self._stop_teleop()
        
        # Stop services
        if self.move_service:
            self.move_service.stop()
        if self.fastsam_service:
            self.fastsam_service.stop()
        if self.grasp_service:
            self.grasp_service.stop()
        if self.service_manager:
            self.service_manager.stop()

        # Close service client (ZMQ REQ socket/context)
        if self.service_client and hasattr(self.service_client, "close"):
            try:
                self.service_client.close()
            except Exception:
                pass

        # Close viewer/env to release render resources immediately
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


def detect():
    """Quick helper to detect objects."""
    assert system.service_client is not None, "Service client not initialized"
    response = system.service_client.call('fastsam_service', {'command': 'GET_LATEST_DETECTION'})
    if response.get('status') == 'success':
        detections = response.get('detections', [])
        print(f"\nDetected {len(detections)} object(s):")
        for i, det in enumerate(detections):
            pc_size = len(det.get('pcd', []))
            pose = det.get('pose', {})
            print(f"  Object {i+1}: {pc_size} points, pos={pose.get('pos', 'N/A')}")
        return detections
    else:
        print(f"Detection failed: {response.get('message')}")
        return []


def get_grasps(detection, num_grasps=5, gripper_type='finger'):
    """Quick helper to generate grasps for a detection."""
    pcd = detection.get('pcd', [])
    pose = detection.get('pose', {})
    
    if not pcd or not pose:
        print("Invalid detection - missing pcd or pose")
        return []
    
    assert system.service_client is not None, "Service client not initialized"
    response = system.service_client.call('grasp_service', {
        'command': 'GET_GRASPS',
        'pcd': pcd,
        'object_pose': pose,
        'num_grasps': num_grasps,
        'gripper_type': gripper_type  # Use simplified type: 'finger' or 'suction'
    })
    
    if response.get('status') == 'success':
        grasps = response.get('grasps', [])
        print(f"\nGenerated {len(grasps)} grasp(s):")
        for i, grasp in enumerate(grasps[:3]):  # Show top 3
            print(f"  Grasp {i+1}: pos={grasp.get('pos')}, score={grasp.get('score', 'N/A'):.3f}")
        return grasps
    else:
        print(f"Grasp generation failed: {response.get('message')}")
        return []

def get_grasps_mesh(pose: StampedPose, object_name: str, num_grasps=5, gripper_type='suction'):
    assert system.service_client is not None, "Service client not initialized"
    response = system.service_client.call('grasp_service', {
        'command': 'GET_GRASPS',
        'object_pose': pose,
        'object_name': object_name,
        'num_grasps': num_grasps,
        'gripper_type': gripper_type  # Use simplified type: 'finger' or 'suction'
    })
    
    if response.get('status') == 'success':
        grasps = response.get('grasps', [])
        print(f"\nGenerated {len(grasps)} grasp(s):")
        for i, grasp in enumerate(grasps[:3]):  # Show top 3
            print(f"  Grasp {i+1}: pos={grasp.get('pos')}, score={grasp.get('score', 'N/A'):.3f}")
        return grasps
    else:
        print(f"Grasp generation failed: {response.get('message')}")
        return []


def get_eef():
    """Quick helper to get end-effector coordinates."""
    if system.tf_manager is None:
        print("TransformManager not initialized")
        return None
        
    try:
        eef_data = {}
        # Iterate over known frames to find EEFs
        # Convention: robot{id}_{arm}_eef
        # We can also iterate robots from env
        if system.env is None:
            print("Environment not initialized")
            return None
            
        for robot_idx, robot in enumerate(system.env.robots):
            robot_name = f"robot{robot_idx}"
            try:
                arm_ids = list(robot.robot_model.eef_name.keys())
            except Exception:
                arm_ids = ["right"] # fallback
            
            for arm_id in arm_ids:
                frame_name = f"{robot_name}_{arm_id}_eef"
                if system.tf_manager.frame_exists(frame_name):
                    pose = system.tf_manager.get_frame_pose(frame_name, "world")
                    eef_data[frame_name] = {
                        "position": {"x": pose.position[0], "y": pose.position[1], "z": pose.position[2]},
                        "quaternion": {"x": pose.quaternion[0], "y": pose.quaternion[1], "z": pose.quaternion[2], "w": pose.quaternion[3]},
                        "frame_id": pose.frame_id
                    }
        
        print("\nEnd-Effector Coordinates:")
        for frame_name, data in eef_data.items():
            pos = data.get('position', {})
            print(f"  {frame_name}: x={pos.get('x', 0):.4f}, y={pos.get('y', 0):.4f}, z={pos.get('z', 0):.4f}")
        return eef_data

    except Exception as e:
        print(f"Failed to get EEF: {e}")
        return None


def set_prompt(prompt):
    """Quick helper to change detection prompt."""
    assert system.service_client is not None, "Service client not initialized"
    response = system.service_client.call('fastsam_service', {
        'command': 'SET_PROMPT',
        'prompt': prompt
    })
    if response.get('status') == 'success':
        print(f"Prompt changed to: '{prompt}'")
    else:
        print(f"Failed to change prompt: {response.get('message')}")

def go_to_pose(pose: StampedPose, robot_id: int, arm_id: str, gripper_action: float = 0.0, approach: bool = False):
    """Quick helper to move an arm to a pose."""
    assert system.service_client is not None, "Service client not initialized"
    
    # StampedPose needs to be serialized to a dict for the service call
    pose_dict = pose.model_dump()
    
    response = system.service_client.call('move_to_pose_service', {
        'command': 'MOVE_TO_POSE',
        'robot_id': robot_id,
        'arm': arm_id,
        'pose': pose_dict,
        'gripper_action': gripper_action,
        'approach': approach
    })
    
    if response.get('status') == 'success':
        print(f"Move command sent to robot {robot_id}, arm {arm_id}")
    else:
        print(f"Failed to send move command: {response.get('message')}")

def cancel_move(robot_id: int, arm_id: str):
    """Quick helper to cancel a move."""
    assert system.service_client is not None, "Service client not initialized"
    response = system.service_client.call('move_to_pose_service', {
        'command': 'CANCEL_MOVE',
        'robot_id': robot_id,
        'arm': arm_id
    })
    
    if response.get('status') == 'success':
        print(f"Cancel command sent to robot {robot_id}, arm {arm_id}")
    else:
        print(f"Failed to send cancel command: {response.get('message')}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Start dual-arm manipulation system with services and interactive shell"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Default detection prompt"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable OpenCV visualization"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        "--meshcat-tf",
        action="store_true",
        help="Stream the environment TF tree to MeshCat (requires `meshcat-server`)"
    )
    
    args = parser.parse_args()
    
    ycb_objects = []

    # Create the dictionary spec for this object
    obj = create_ycb_object(
        ycb_name="plate",
    )
    ycb_objects.append(obj)

    # Create and start system
    global system
    system = DualArmManipulationSystem(
        visualize=args.visualize,
        debug=args.debug,
        detection_prompt=args.prompt,
        meshcat_tf=args.meshcat_tf,
        ycb_objects=ycb_objects,
    )

    # Ensure cleanup on normal interpreter shutdown and on SIGINT/SIGTERM
    atexit.register(lambda: system.stop())

    def _handle_signal(signum: int, _frame: Any) -> None:
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)
        system.request_shutdown(reason=f"signal {signame}")

    try:
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
    except Exception:
        pass
    
    try:
        system.start(start_teleop_thread=False)
        
        # Wait a moment for everything to stabilize
        time.sleep(1.0)
        
        # Create interactive shell (run in separate thread so teleop can run in main thread)
        banner = """
Interactive Python Shell - Dual Arm Manipulation System
========================================================

Available objects:
  client          - ServiceClient for direct service calls
  system          - DualArmManipulationSystem instance
  
Helper functions:
  detect()                               - Detect objects
  get_grasps(detection, n=5)             - Generate grasps for detection
  get_eef()                              - Get end-effector coordinates
  set_prompt(text)                       - Change detection prompt
  go_to_pose(pose, robot_id, arm_id)     - Move an arm to a pose
  cancel_move(robot_id, arm_id)          - Cancel a move command

Example usage:
  >>> detections = detect()
  >>> if detections:
  ...     grasps = get_grasps(detections[0])
  >>> eef_pose = get_eef()['robot0_right_eef'] # get a pose
  >>> go_to_pose(StampedPose(position=[eef_pose['position']['x'], eef_pose['position']['y']+0.1, eef_pose['position']['z']], quaternion=eef_pose['quaternion'], frame_id='world'), 0, 'right')
  >>> cancel_move(0, 'right')

Press Ctrl+D or type exit() to quit
========================================================
"""
        
        client = system.service_client

        def shell_thread():
            try:
                # Create namespace with system objects and helper functions
                namespace = {
                    'client': client,
                    'system': system,
                    'detect': detect,
                    'get_grasps': get_grasps,
                    'get_grasps_mesh': get_grasps_mesh,
                    'get_eef': get_eef,
                    'set_prompt': set_prompt,
                    'go_to_pose': go_to_pose,
                    'cancel_move': cancel_move,
                    'env_step': env_step,
                    'StampedPose': StampedPose,
                }
                code.interact(banner=banner, local=namespace)
            except SystemExit:
                pass

        t = threading.Thread(target=shell_thread, daemon=True)
        t.start()

        # Run teleoperation in main thread to ensure viewer renders
        system.run_teleop(blocking=True)

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 0
    except Exception:
        logger.exception("Fatal error while starting/running the system", exc_info=True)
        return 1
    finally:
        system.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
