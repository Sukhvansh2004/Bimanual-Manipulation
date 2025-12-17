# Robosuite Franka Handoff

This package provides a robosuite-based implementation of dual-arm Franka Panda robot control for handoff tasks, ported from ROS.

## Structure

- `controllers/`: Control classes for arms and grippers
  - `arm.py`: RobosuiteArmController for moving arms to poses
  - `gripper.py`: RobosuiteGripperController for gripper actions
- `perception/`: Object detection and grasp prediction
  - `object_detector.py`: Detects objects in the scene
  - `grasp_predictor.py`: Predicts grasp poses using GraspGen
- `scripts/`: Executable scripts
  - `testing/test_controllers.py`: Tests environment setup and controller functionality
  - `manipulation/dual_arm_manipulation.py`: Full dual-arm manipulation with perception
  - `manipulation/handoff_sequence.py`: Runs the full handoff sequence with perception
  - `demos/go_to_pose.py`: Sends pose goals to arms
  - `demos/gripper_move.py`: Tests gripper open/close
  - `teleop/teleop_dual_arm.py`: Manual teleoperation of dual arms with keyboard
  - `gripper_move.py`: Tests gripper open/close

## Features

- **Perception Integration**: Automatically detects objects and predicts grasps using GraspGen
- **Dual-Arm Control**: Coordinated control of two Franka Panda arms
- **Handoff Sequence**: Complete pick-and-place with handoff between arms

## Usage

To test the system:

```bash
python -m robosuite_franka_handoff.scripts.testing.test_controllers
```

To run dual-arm manipulation with perception:

```bash
python -m robosuite_franka_handoff.scripts.manipulation.dual_arm_manipulation --task handoff
```

Or for cooperative lifting:

```bash
python -m robosuite_franka_handoff.scripts.manipulation.dual_arm_manipulation --task cooperative_lift
```

To run the handoff sequence with perception:

```bash
python -m robosuite_franka_handoff.scripts.manipulation.handoff_sequence
```

For teleoperation:

```bash
python -m robosuite_franka_handoff.scripts.teleop.teleop_dual_arm
```

Or from within the package:

```python
from robosuite_franka_handoff import make_dual_arm_handoff_env, ObjectDetector, GraspPredictor
```

## Requirements

- robosuite
- numpy
- scipy
- GraspGen (for grasp prediction, optional)