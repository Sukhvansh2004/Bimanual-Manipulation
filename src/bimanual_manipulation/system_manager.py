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
from typing import Any, Optional

from scipy.spatial.transform import Rotation as R

from robosuite.wrappers import VisualizationWrapper
import bimanual_manipulation.constants as constants
from bimanual_manipulation.environments.dual_arm import make_dual_arm_env, DualArmEnvironment
try:
    from bimanual_manipulation.teleop.teleop import KeyboardTeleop
except ImportError:
    KeyboardTeleop = None
from bimanual_manipulation.controllers.arm import RobosuiteArmController
from bimanual_manipulation.controllers.gripper import RobosuiteGripperController
from bimanual_manipulation.services.fastsam_service import FastSAMService
from bimanual_manipulation.services.grasp_service import GraspService
from bimanual_manipulation.services.move_to_pose_service import MoveToPoseService
from bimanual_manipulation.services.services_core import ServiceManager, ServiceClient
from bimanual_manipulation.transforms.stamped_pose import StampedPose, Transform
from bimanual_manipulation.transforms.transform_manager import TransformManager
from bimanual_manipulation.transforms.meshcat_tf_visualizer import (
    MeshcatTFVisualizerConfig,
    stream_transforms_to_meshcat,
)
from bimanual_manipulation.utils.env_utils import env_step, set_robot_config

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
            table_full_size: Optional[Any] = None,
            table_offset: Optional[Any] = None,
            robot_base_configs: Optional[Any] = None,
            start_services: bool = True,
            server_host: Optional[str] = None,
            server_port: int = constants.SERVICE_MANAGER_PORT,
            ):
        
        self.custom_objects = custom_objects
        self.ycb_objects = ycb_objects
        self.table_full_size = table_full_size
        self.table_offset = table_offset
        self.robot_base_configs = robot_base_configs
        self.visualize = visualize
        self.debug = debug
        self.detection_prompt = detection_prompt
        self.meshcat_tf = meshcat_tf
        self.use_fastsam = use_fastsam
        self.use_grasp_service = use_grasp_service
        self.has_renderer = has_renderer
        self.start_services = start_services
        self.server_host = server_host
        self.server_port = server_port
        # Use RLock to prevent deadlocks (TransformManager re-acquires it)
        self.env_lock = threading.RLock()

        if self.debug:
            logger.setLevel(logging.DEBUG)

        self.env: DualArmEnvironment = None
        self.planning_env: Optional[DualArmEnvironment] = None
        self.tf_manager: Optional[TransformManager] = None
        self.teleop: Optional[KeyboardTeleop] = None # type: ignore
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
        
        # Create environments
        logger.info("\n1. Creating dual-arm environment...")
        self.env = make_dual_arm_env(
            custom_objects=self.custom_objects, 
            ycb_objects=self.ycb_objects, 
            has_renderer=self.has_renderer,
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
            robot_base_configs=self.robot_base_configs,
        )
        if self.has_renderer:
            logger.debug("Adding VisualizationWrapper to environment...")
            self.env = VisualizationWrapper(self.env)
        
        # Create a separate headless environment for planning (the "Twin Env")
        # This prevents wild jumping behavior in the viewer during RRT planning.
        logger.info("   Creating separate planning environment (headless)...")
        self.planning_env = make_dual_arm_env(
            custom_objects=self.custom_objects,
            ycb_objects=self.ycb_objects,
            has_renderer=False,
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
            robot_base_configs=self.robot_base_configs,
        )
        logger.info("   ✓ Environments created")

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
        if self.start_services:
            logger.info("\n2. Starting services...")
            self._start_services()
        else:
            logger.info("\n2. Services disabled — skipping.")
        
        # Prepare teleoperation
        logger.info("\n3. Preparing teleoperation...")
        if KeyboardTeleop is not None:
            self.teleop = KeyboardTeleop(self.env, env_lock=self.env_lock)
            if start_teleop_thread:
                self.run_teleop(blocking=False)
                logger.info("   ✓ Teleoperation started (thread)")
            else:
                logger.info("   Teleoperation ready. Run system.run_teleop() to start.")
        else:
            logger.info("   Teleoperation unavailable (no display). Skipping.")
        
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

    def sync_planning_env(self) -> None:
        """Synchronize the planning environment state with the real environment.
        
        Copies all joint positions (qpos), velocities (qvel), and equality 
        constraints (welds) to ensure that planning correctly reflects 
        the current system state.
        """
        if self.env is None or self.planning_env is None:
            return
        
        with self.env_lock:
            # Sync core state
            self.planning_env.sim.data.qpos[:] = self.env.sim.data.qpos
            self.planning_env.sim.data.qvel[:] = self.env.sim.data.qvel
            
            # Sync equality constraints (e.g. suction welds)
            # Both model (targets) and data (active state) must be synced
            self.planning_env.sim.model.eq_obj1id[:] = self.env.sim.model.eq_obj1id
            self.planning_env.sim.model.eq_obj2id[:] = self.env.sim.model.eq_obj2id
            self.planning_env.sim.model.eq_data[:] = self.env.sim.model.eq_data
            self.planning_env.sim.data.eq_active[:] = self.env.sim.data.eq_active
            
            self.planning_env.sim.forward()
        logger.debug("Planning environment synchronized with real environment (including welds).")

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
        """Start services.
        
        In client-only mode (server_host is set), connects to a remote
        ServiceManager instead of starting one locally. GPU-bound services
        (FastSAM, Grasp) are assumed to run on the remote server.
        Only the local MoveToPose service is started.
        """
        try:
            if self.server_host:
                # Client-only mode: connect to remote ServiceManager
                logger.info("   Client-only mode: connecting to remote server at %s:%d...",
                            self.server_host, self.server_port)
                self.service_client = ServiceClient(
                    manager_host=self.server_host,
                    manager_port=self.server_port,
                )
                logger.info("   ✓ Connected to remote Service Manager")
            else:
                # Full mode: start local ServiceManager + GPU services
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
            
            # Start Move To Pose service (always local — needs env access)
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

        if self.planning_env is not None:
            try:
                if hasattr(self.planning_env, "close"):
                    self.planning_env.close()
            except Exception:
                pass
        
        self.running = False
        logger.info("System stopped")
