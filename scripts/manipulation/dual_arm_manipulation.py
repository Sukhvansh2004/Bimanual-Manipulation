#!/usr/bin/env python3
"""
Interactive Teleoperation with Service Integration

This script provides an interactive teleoperation interface that allows you to:
- Teleoperate dual-arm robots with keyboard controls
- Call FastSAM service for object detection
- Call GraspGen service for grasp prediction
- Query EEF service for end-effector coordinates (debugging)
- Execute grasp actions on detected objects
- Visualize detections and grasps in real-time

All services communicate through the ServiceManager for proper abstraction.

Key bindings:
    Robosuite standard controls (arrow keys, etc.) - Move robot
    'd' - Detect objects using FastSAM
    'g' - Generate grasps for detected objects
    'x' - Execute grasp on first detected object
    'v' - Toggle visualization
    'p' - Change detection prompt
    'e' - Get end-effector coordinates (debug)
    'a' - Exit teleoperation

Usage:
    python scripts/manipulation/dual_arm_manipulation.py
    python scripts/manipulation/dual_arm_manipulation.py --prompt "red cup"
    python scripts/manipulation/dual_arm_manipulation.py --visualize
"""

import argparse
import numpy as np
import zmq
import time
import logging
import copy
from typing import Dict, Any, List, Optional
from pynput import keyboard as pynput_keyboard

import robosuite as suite
from robosuite.wrappers import VisualizationWrapper

# Import custom modules
from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.teleop import KeyboardTeleop
from robosuite_franka_manipulation.services.fastsam_service import FastSAMService
from robosuite_franka_manipulation.services.grasp_service import GraspService
from robosuite_franka_manipulation.services.eef_service import EEFService
from robosuite_franka_manipulation.services.service_manager import ServiceManager, ServiceClient
from robosuite_franka_manipulation import constants

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InteractiveTeleopWithServices:
    """
    Interactive teleoperation with integrated perception and grasp services.
    """

    def __init__(self, 
                 env,
                 pos_sensitivity: float = 1.0,
                 rot_sensitivity: float = 1.0,
                 detection_prompt: str = "object",
                 visualize: bool = False,
                 auto_start_services: bool = True):
        """
        Initialize interactive teleoperation.

        Args:
            env: Robosuite environment
            pos_sensitivity: Position control sensitivity
            rot_sensitivity: Rotation control sensitivity
            detection_prompt: Default prompt for object detection
            visualize: Enable OpenCV visualization
            auto_start_services: Automatically start FastSAM and Grasp services
        """
        self.env = env
        self.visualize = visualize
        self.detection_prompt = detection_prompt
        
        # Initialize teleoperation
        self.teleop = KeyboardTeleop(env, pos_sensitivity, rot_sensitivity)
        
        # Service state
        self.services_running = False
        self.service_manager = None
        self.fastsam_service = None
        self.grasp_service = None
        self.eef_service = None
        self.service_client = None
        
        # Detection state
        self.latest_detections = []
        self.latest_grasps = []
        
        # Additional key bindings for services
        self.service_key_pressed = {
            'd': False,  # Detect objects
            'g': False,  # Generate grasps
            'x': False,  # Execute grasp
            'v': False,  # Toggle visualization
            'p': False,  # Change prompt
            'e': False,  # Get EEF coordinates
        }
        
        # Setup additional keyboard listener for service keys
        self.service_listener = pynput_keyboard.Listener(on_press=self._on_service_key_press)
        self.service_listener.start()
        
        # Auto-start services if requested
        if auto_start_services:
            self.start_services()
        
        logger.info("Interactive Teleoperation initialized")
        self._print_help()

    def _on_service_key_press(self, key):
        """Callback for service-related key presses."""
        try:
            if hasattr(key, 'char'):
                char = key.char
                if char in self.service_key_pressed:
                    self.service_key_pressed[char] = True
        except AttributeError:
            pass

    def _print_help(self):
        """Print help information."""
        print("\n" + "="*70)
        print("Interactive Teleoperation with Services")
        print("="*70)
        print("\nRobosuite Controls:")
        print("  Arrow keys       - Move horizontally in x-y plane")
        print("  .-;              - Move vertically")
        print("  o-p              - Rotate (yaw)")
        print("  y-h              - Rotate (pitch)")
        print("  e-r              - Rotate (roll)")
        print("  spacebar         - Toggle gripper (open/close)")
        print("  s                - Switch active arm")
        print("  Ctrl+q           - Reset simulation")
        print("\nService Controls:")
        print("  d                - Detect objects (FastSAM)")
        print("  g                - Generate grasps (GraspGen)")
        print("  x                - Execute grasp on first object")
        print("  v                - Toggle visualization")
        print("  p                - Change detection prompt")
        print("  e                - Get end-effector coordinates (debug)")
        print("  a                - Exit teleoperation")
        print("="*70 + "\n")

    def start_services(self):
        """Start FastSAM and Grasp services."""
        if self.services_running:
            logger.warning("Services already running")
            return

        try:
            logger.info("Starting perception services...")
            
            # Start Service Manager first
            logger.info("Starting Service Manager...")
            self.service_manager = ServiceManager()
            self.service_manager.start()
            time.sleep(0.5)  # Give manager time to start
            logger.info("✓ Service Manager started")
            
            # Initialize service client
            self.service_client = ServiceClient()
            
            # Start FastSAM service
            logger.info("Initializing FastSAM service...")
            self.fastsam_service = FastSAMService(
                env=self.env,
                visualize=self.visualize,
                debug=False
            )
            self.fastsam_service.start()
            time.sleep(0.5)
            logger.info("✓ FastSAM service started")
            
            # Set initial prompt via ServiceManager
            self._set_detection_prompt(self.detection_prompt)
            
            # Start Grasp service
            logger.info("Initializing Grasp service...")
            self.grasp_service = GraspService(
                debug=False,
                visualize=self.visualize
            )
            self.grasp_service.start()
            time.sleep(0.5)
            logger.info("✓ Grasp service started")
            
            # Start EEF service for debugging
            logger.info("Initializing EEF service...")
            self.eef_service = EEFService(env=self.env)
            self.eef_service.start()
            time.sleep(0.5)
            logger.info("✓ EEF service started")
            
            self.services_running = True
            logger.info("All services started successfully!\n")
            
        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            self.services_running = False

    def stop_services(self):
        """Stop all services."""
        if not self.services_running:
            return

        logger.info("Stopping services...")
        
        if self.eef_service:
            self.eef_service.stop()
        
        if self.fastsam_service:
            self.fastsam_service.stop()
        
        if self.grasp_service:
            self.grasp_service.stop()
        
        if self.service_manager:
            self.service_manager.stop()
        
        self.services_running = False
        logger.info("Services stopped")

    def _set_detection_prompt(self, prompt: str):
        """Update detection prompt in FastSAM service via ServiceManager."""
        try:
            if not self.service_client:
                logger.warning("Service client not initialized")
                return
                
            response = self.service_client.call(
                'fastsam_service',
                {'command': 'SET_PROMPT', 'prompt': prompt}
            )
            
            if response.get('status') == 'success':
                self.detection_prompt = prompt
                logger.info(f"Detection prompt set to: '{prompt}'")
            else:
                logger.warning(f"Failed to set prompt: {response.get('message')}")
            
        except Exception as e:
            logger.error(f"Failed to set prompt: {e}")

    def _detect_objects(self):
        """Trigger object detection via ServiceManager."""
        if not self.services_running:
            logger.warning("Services not running. Start services first.")
            return

        logger.info(f"Detecting objects with prompt: '{self.detection_prompt}'...")
        
        try:
            # Query FastSAM service via ServiceManager
            response = self.service_client.call(
                'fastsam_service',
                {'command': 'GET_LATEST_DETECTION'}
            )
            
            if response.get('status') == 'success':
                detections = response.get('detections', [])
                self.latest_detections = detections
                
                logger.info(f"✓ Detected {len(detections)} object(s)")
                
                for i, det in enumerate(detections):
                    pcd = det.get('pointcloud', [])
                    pose = det.get('pose', {})
                    logger.info(f"  Object {i+1}: {len(pcd)} points, pose={pose.get('pos', 'N/A')}")
            else:
                logger.warning(f"Detection failed: {response.get('message')}")
                self.latest_detections = []
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            self.latest_detections = []

    def _generate_grasps(self, num_grasps: int = 5):
        """Generate grasps for detected objects via ServiceManager."""
        if not self.services_running:
            logger.warning("Services not running. Start services first.")
            return

        if not self.latest_detections:
            logger.warning("No objects detected. Run detection first (press 'd')")
            return

        logger.info(f"Generating {num_grasps} grasps for {len(self.latest_detections)} object(s)...")
        
        try:
            self.latest_grasps = []
            
            # Query Grasp service for each detection via ServiceManager
            for i, detection in enumerate(self.latest_detections):
                pointcloud = detection.get('pointcloud', [])
                pose = detection.get('pose', {})
                
                if not pointcloud or not pose:
                    logger.warning(f"  Object {i+1}: Missing pointcloud or pose")
                    continue
                
                # Call grasp service via ServiceManager
                request = {
                    'command': 'GET_GRASPS',
                    'pointcloud': pointcloud if isinstance(pointcloud, list) else pointcloud.tolist(),
                    'object_pose': pose,
                    'num_grasps': num_grasps,
                    'gripper_type': constants.DEFAULT_GRIPPER_TYPE
                }
                
                response = self.service_client.call('grasp_service', request)
                
                if response.get('status') == 'success':
                    grasps = response.get('grasps', [])
                    self.latest_grasps.extend(grasps)
                    logger.info(f"  ✓ Object {i+1}: Generated {len(grasps)} grasp(s)")
                    
                    # Show top grasp
                    if grasps:
                        top_grasp = grasps[0]
                        logger.info(f"    Top grasp: pos={top_grasp.get('pos')}, score={top_grasp.get('score', 'N/A')}")
                else:
                    logger.warning(f"  Object {i+1}: Grasp generation failed - {response.get('message')}")
            
            logger.info(f"✓ Total {len(self.latest_grasps)} grasp(s) generated")
            
        except Exception as e:
            logger.error(f"Grasp generation error: {e}")

    def _execute_grasp(self):
        """Execute grasp on the first detected object."""
        if not self.latest_grasps:
            logger.warning("No grasps available. Generate grasps first (press 'g')")
            return

        logger.info("Executing grasp on first object...")
        
        # Get the top-ranked grasp
        grasp = self.latest_grasps[0]
        grasp_pos = np.array(grasp.get('pos', [0, 0, 0]))
        grasp_quat = np.array(grasp.get('quat', [1, 0, 0, 0]))
        
        logger.info(f"Target grasp pose: pos={grasp_pos}, quat={grasp_quat}")
        logger.info("Note: Actual execution requires trajectory planning integration")
        logger.info("      Use the trajectory planning controller to execute this grasp")
        
        # TODO: Integrate with trajectory planning to actually move the robot
        # This would require:
        # 1. Plan trajectory to pre-grasp pose
        # 2. Move to grasp pose
        # 3. Close gripper
        # 4. Lift object

    def _change_prompt(self):
        """Interactive prompt change."""
        print("\nCurrent prompt:", self.detection_prompt)
        print("Enter new prompt (or press Enter to keep current):")
        # Note: This is a simplified version. In practice, you'd need non-blocking input
        # For now, just cycle through common prompts
        prompts = ["object", "cup", "bottle", "box", "toy", "tool"]
        current_idx = prompts.index(self.detection_prompt) if self.detection_prompt in prompts else 0
        new_prompt = prompts[(current_idx + 1) % len(prompts)]
        self._set_detection_prompt(new_prompt)

    def _toggle_visualization(self):
        """Toggle visualization."""
        self.visualize = not self.visualize
        logger.info(f"Visualization: {'ON' if self.visualize else 'OFF'}")

    def _get_eef_coordinates(self):
        """Query EEF service for current end-effector coordinates (debug)."""
        if not self.services_running:
            logger.warning("Services not running. Start services first.")
            return

        logger.info("Querying end-effector coordinates...")
        
        try:
            # Query EEF service via ServiceManager
            response = self.service_client.call(
                'eef_service',
                {'command': 'GET'}
            )
            
            if response.get('success'):
                eef_data = response.get('eef_coordinates', {})
                logger.info("✓ End-Effector Coordinates:")
                
                for robot_key, data in eef_data.items():
                    pos = data.get('position', {})
                    orient = data.get('orientation', {})
                    logger.info(f"  {robot_key}:")
                    logger.info(f"    Position: x={pos.get('x', 0):.4f}, y={pos.get('y', 0):.4f}, z={pos.get('z', 0):.4f}")
                    logger.info(f"    Quaternion: w={orient.get('w', 0):.4f}, x={orient.get('x', 0):.4f}, y={orient.get('y', 0):.4f}, z={orient.get('z', 0):.4f}")
            else:
                logger.warning(f"Failed to get EEF coordinates: {response.get('error')}")
            
        except Exception as e:
            logger.error(f"EEF query error: {e}")

    def _check_service_keys(self):
        """Check and handle service key presses."""
        # Detect objects
        if self.service_key_pressed['d']:
            self.service_key_pressed['d'] = False
            self._detect_objects()
        
        # Generate grasps
        if self.service_key_pressed['g']:
            self.service_key_pressed['g'] = False
            self._generate_grasps()
        
        # Execute grasp
        if self.service_key_pressed['x']:
            self.service_key_pressed['x'] = False
            self._execute_grasp()
        
        # Toggle visualization
        if self.service_key_pressed['v']:
            self.service_key_pressed['v'] = False
            self._toggle_visualization()
        
        # Change prompt
        if self.service_key_pressed['p']:
            self.service_key_pressed['p'] = False
            self._change_prompt()
        
        # Get EEF coordinates
        if self.service_key_pressed['e']:
            self.service_key_pressed['e'] = False
            self._get_eef_coordinates()

    def run(self):
        """Run the interactive teleoperation loop."""
        logger.info("Starting interactive teleoperation...")
        
        # Modify teleop to check for service keys
        original_run = self.teleop.run
        
        # Override the teleop run to add service key checking
        self.teleop.running = True
        print("Starting teleoperation with robosuite Keyboard device.")
        print("Controls: Follow robosuite's keyboard controls.")
        print("Press 'a' key to exit teleoperation.")

        while self.teleop.running:
            # Reset the environment
            obs = self.env.reset()
            self.env.render()

            # Initialize device control
            self.teleop.device.start_control()

            # Initialize variables that should be maintained between resets
            all_prev_gripper_actions = [
                {
                    f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                    for robot_arm in robot.arms
                    if robot.gripper[robot_arm].dof > 0
                }
                for robot in self.env.robots
            ]

            # Loop until we get a reset from the input or the task completes
            while True:
                # Check for service key presses
                self._check_service_keys()
                
                # Set active robot
                active_robot = self.env.robots[self.teleop.device.active_robot]

                # Get the newest action
                input_ac_dict = self.teleop.device.input2action(mirror_actions=False)

                # If action is none, then this a reset so we should break
                if input_ac_dict is None:
                    break

                action_dict = copy.deepcopy(input_ac_dict)
                # set arm actions
                for arm in active_robot.arms:
                    # OSC controller is now set to delta mode for Keyboard
                    action_dict[arm] = input_ac_dict[f"{arm}_delta"]

                # Maintain gripper state for each robot but only update the active robot with action
                env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(self.env.robots)]
                env_action[self.teleop.device.active_robot] = active_robot.create_action_vector(action_dict)
                env_action = np.concatenate(env_action)

                # Update gripper state for the active robot from the current action
                for gripper_ac in all_prev_gripper_actions[self.teleop.device.active_robot]:
                    all_prev_gripper_actions[self.teleop.device.active_robot][gripper_ac] = action_dict[gripper_ac]

                # Step through the environment
                self.env.step(env_action)
                self.env.render()

                # Check for exit key press
                if self.teleop.exit_pressed:
                    self.teleop.running = False
                    break

        # Cleanup
        self.teleop.listener.stop()
        self.service_listener.stop()
        self.stop_services()
        print("Teleoperation ended.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive teleoperation with FastSAM and GraspGen services"
    )
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="Position control sensitivity"
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="Rotation control sensitivity"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="object",
        help="Default detection prompt for FastSAM"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable OpenCV visualization windows"
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Don't automatically start services"
    )
    
    args = parser.parse_args()

    # Create environment
    logger.info("Creating dual-arm environment...")
    env = make_dual_arm_env()
    env = VisualizationWrapper(env)
    logger.info("✓ Environment created")

    # Create interactive teleop
    interactive_teleop = InteractiveTeleopWithServices(
        env=env,
        pos_sensitivity=args.pos_sensitivity,
        rot_sensitivity=args.rot_sensitivity,
        detection_prompt=args.prompt,
        visualize=args.visualize,
        auto_start_services=not args.no_auto_start
    )

    # Add keypress callback
    env.viewer.add_keypress_callback(interactive_teleop.teleop.device.on_press)

    # Run
    try:
        interactive_teleop.run()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    finally:
        interactive_teleop.stop_services()


if __name__ == "__main__":
    main()
