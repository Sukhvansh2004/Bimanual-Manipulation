#!/usr/bin/env python3
"""
Start Services and Interactive Teleoperation

This script:
1. Starts the dual-arm environment with teleoperation in a separate thread
2. Starts all services (ServiceManager, FastSAM, GraspGen, EEF)
3. Provides an interactive Python shell for service queries

You can teleoperate the robot in the environment window using standard robosuite
keyboard controls, while simultaneously using the Python terminal to:
- Query detections: client.call('fastsam_service', {'command': 'GET_LATEST_DETECTION'})
- Generate grasps: client.call('grasp_service', {'command': 'GET_GRASPS', ...})
- Get EEF coords: client.call('eef_service', {'command': 'GET'})
- Change prompts: client.call('fastsam_service', {'command': 'SET_PROMPT', 'prompt': 'cup'})

Usage:
    python scripts/manipulation/start_services_and_teleop.py
    
    Then in the interactive shell:
    >>> # Get latest detections
    >>> response = client.call('fastsam_service', {'command': 'GET_LATEST_DETECTION'})
    >>> print(response)
    
    >>> # Change detection prompt
    >>> client.call('fastsam_service', {'command': 'SET_PROMPT', 'prompt': 'red cup'})
    
    >>> # Get end-effector coordinates
    >>> client.call('eef_service', {'command': 'GET'})
    
    >>> # Generate grasps (after detecting objects)
    >>> detections = response['detections']
    >>> if detections:
    ...     pc = detections[0]['pointcloud']
    ...     pose = detections[0]['pose']
    ...     grasps = client.call('grasp_service', {
    ...         'command': 'GET_GRASPS',
    ...         'pointcloud': pc,
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
from robosuite_franka_manipulation.services.eef_service import EEFService
from robosuite_franka_manipulation.services.service_manager import ServiceManager, ServiceClient
from robosuite_franka_manipulation.transforms import TransformManager, MeshcatTFVisualizerConfig, stream_transforms_to_meshcat

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
    
    def __init__(self, visualize: bool = False, debug: bool = False, detection_prompt: str = "object", meshcat_tf: bool = False):
        self.visualize = visualize
        self.debug = debug
        self.detection_prompt = detection_prompt
        self.meshcat_tf = meshcat_tf
        self.env_lock = threading.Lock()

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
        self.eef_service: Optional[EEFService] = None
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
        self.env = make_dual_arm_env()
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
            self.env, arm_id="right", robot_id=0, env_lock=self.env_lock, transform_manager=self.tf_manager
        )
        self.right_arm = RobosuiteArmController(
            self.env, arm_id="right", robot_id=1, env_lock=self.env_lock, transform_manager=self.tf_manager
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
        self.teleop = KeyboardTeleop(self.env)
        if start_teleop_thread:
            self.teleop_thread = threading.Thread(target=self._run_teleop_loop, daemon=True)
            self.teleop_thread.start()
            logger.info("   ✓ Teleoperation started (thread)")
        else:
            logger.info("   Teleoperation ready. Run system.run_teleop_blocking() to start in main thread.")
        
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
            
            # Start EEF service
            logger.info("   Starting EEF service...")
            self.eef_service = EEFService(env=self.env, env_lock=self.env_lock)
            self.eef_service.start()
            time.sleep(0.5)
            logger.info("   ✓ EEF service started")
            
        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            raise
    
    def _run_teleop_loop(self):
        """Run teleoperation in separate thread."""
        try:
            # Add keypress callback
            assert self.env is not None, "Environment not initialized"
            assert self.teleop is not None, "Teleop not initialized"
            self.env.viewer.add_keypress_callback(self.teleop.device.on_press)  # type: ignore[union-attr]
            
            # Run teleoperation guarded by env_lock
            with self.env_lock:
                pass  # ensure lock exists
            self.teleop.run(env_lock=self.env_lock)
            
        except Exception as e:
            logger.error(f"Teleoperation error: {e}")

    def run_teleop_blocking(self):
        """Run teleoperation in the current (main) thread with env lock."""
        # Add keypress callback
        assert self.env is not None, "Environment not initialized"
        assert self.teleop is not None, "Teleop not initialized"
        self.env.viewer.add_keypress_callback(self.teleop.device.on_press)

        # Also handle quit in the viewer callback so 'q' works reliably when the
        # mujoco window has focus and we can shut down services immediately.
        def _on_viewer_keypress(key: Any) -> None:
            if self._is_quit_key(key):
                # Stop everything immediately (services + viewer)
                self.request_shutdown(reason="q pressed")

        try:
            self.env.viewer.add_keypress_callback(_on_viewer_keypress)
        except Exception:
            # Viewer implementations differ; teleop still listens via pynput.
            pass
        self.teleop.run(env_lock=self.env_lock)
    
    def stop(self):
        """Stop the system."""
        with self._stop_lock:
            if not self.running and not self._shutdown_requested.is_set():
                return
            
        logger.info("\nStopping system...")
        
        # Stop teleop
        if self.teleop:
            try:
                self.teleop.exit_pressed = True
                self.teleop.running = False
                if hasattr(self.teleop, "listener") and self.teleop.listener is not None:
                    self.teleop.listener.stop()
            except Exception:
                pass
        
        # Stop services
        if self.eef_service:
            self.eef_service.stop()
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
            pc_size = len(det.get('pointcloud', []))
            pose = det.get('pose', {})
            print(f"  Object {i+1}: {pc_size} points, pos={pose.get('pos', 'N/A')}")
        return detections
    else:
        print(f"Detection failed: {response.get('message')}")
        return []


def get_grasps(detection, num_grasps=5, gripper_type='finger'):
    """Quick helper to generate grasps for a detection."""
    pointcloud = detection.get('pointcloud', [])
    pose = detection.get('pose', {})
    
    if not pointcloud or not pose:
        print("Invalid detection - missing pointcloud or pose")
        return []
    
    assert system.service_client is not None, "Service client not initialized"
    response = system.service_client.call('grasp_service', {
        'command': 'GET_GRASPS',
        'pointcloud': pointcloud,
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


def get_eef():
    """Quick helper to get end-effector coordinates."""
    assert system.service_client is not None, "Service client not initialized"
    response = system.service_client.call('eef_service', {'command': 'GET'})
    if response.get('success'):
        eef_data = response.get('eef_coordinates', {})
        print("\nEnd-Effector Coordinates:")
        for robot_key, data in eef_data.items():
            pos = data.get('position', {})
            print(f"  {robot_key}: x={pos.get('x', 0):.4f}, y={pos.get('y', 0):.4f}, z={pos.get('z', 0):.4f}")
        return eef_data
    else:
        print(f"Failed to get EEF: {response.get('error')}")
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


def main():
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
    
    # Create and start system
    global system
    system = DualArmManipulationSystem(
        visualize=args.visualize,
        debug=args.debug,
        detection_prompt=args.prompt,
        meshcat_tf=args.meshcat_tf,
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
  detect()                    - Detect objects
  get_grasps(detection, n=5)  - Generate grasps for detection
  get_eef()                   - Get end-effector coordinates
  set_prompt(text)            - Change detection prompt

Example usage:
  >>> detections = detect()
  >>> if detections:
  ...     grasps = get_grasps(detections[0])
  >>> get_eef()
  >>> set_prompt('red cup')

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
                    'get_eef': get_eef,
                    'set_prompt': set_prompt,
                }
                code.interact(banner=banner, local=namespace)
            except SystemExit:
                pass

        t = threading.Thread(target=shell_thread, daemon=True)
        t.start()

        # Run teleoperation in main thread to ensure viewer renders
        system.run_teleop_blocking()

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    finally:
        # If we reached here, try a graceful stop first.
        system.stop()
        # Then ensure the process exits promptly.
        sys.exit(0)


if __name__ == "__main__":
    main()
