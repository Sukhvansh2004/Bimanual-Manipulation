# TODO: Switch from linear translation → minimum jerk trajectories.

import numpy as np
from typing import List, Dict, Union, Callable, Any, Tuple
from scipy.spatial.transform import Rotation as R, Slerp

from robosuite_franka_manipulation.transforms import StampedPose


# Utilities
def resolve_axis(axis: Union[str, np.ndarray, Callable[[np.ndarray], np.ndarray]], quaternion: np.ndarray) -> np.ndarray:
    """
    Resolve a motion axis.

    axis options:
        'x','y','z','-x','-y','-z'  -> EE frame
        np.ndarray(3,)              -> world frame
        callable(quaternion)->axis  -> custom
    """
    # Convert quaternion to rotation matrix for axis resolution
    rot = R.from_quat(quaternion)
    rotation = rot.as_matrix()

    if callable(axis):
        vec = axis(rotation)

    elif isinstance(axis, str):
        mapping = {
            'x': rotation[:, 0],
            'y': rotation[:, 1],
            'z': rotation[:, 2],
            '-x': -rotation[:, 0],
            '-y': -rotation[:, 1],
            '-z': -rotation[:, 2],
        }

        if axis.lower() not in mapping:
            raise ValueError(f"Invalid axis string: {axis}")

        vec = mapping[axis.lower()]

    else:
        vec = np.asarray(axis)

    norm = np.linalg.norm(vec)

    if norm < 1e-8:
        raise ValueError("Axis has near-zero magnitude.")

    return vec / norm


def slerp_quat(q1: np.ndarray, q2: np.ndarray, alpha: float) -> np.ndarray:
    """
    Proper spherical interpolation between quaternions.
    Produces constant angular velocity.
    """
    key_times = [0, 1]
    rotations = R.from_quat([q1, q2])
    slerp_obj = Slerp(key_times, rotations)

    return slerp_obj([alpha]).as_quat()[0]


def interpolate_pose(p1: StampedPose, p2: StampedPose, steps: int) -> List[StampedPose]:
    """
    Interpolate SE(3) using linear translation + SLERP rotation.
    """
    traj = []

    for i in range(steps):
        a = i / (steps - 1) if steps > 1 else 0.0

        pos = (1 - a) * p1.position + a * p2.position
        quat = slerp_quat(p1.quaternion, p2.quaternion, a)

        traj.append(StampedPose(position=pos, quaternion=quat, frame_id=p1.frame_id))

    return traj


def sigmoid_ramp(steps: int) -> np.ndarray:
    x = np.linspace(-6, 6, steps)
    s = 1 / (1 + np.exp(-x))
    return (s - s.min()) / (s.max() - s.min())


class DualArmHandoverPlanner:

    def __init__(
        self,
        grasp_transfer: StampedPose,
        grasp_receiver: StampedPose,
        handover_object_pose: StampedPose,
        pre_align_distance: float = 0.20,
        pre_align_axis: str = '-z',
        retreat_distance: float = 0.05,
        approach_steps: int = 50,
        ramp_steps: int = 60,
        transfer_retreat_axis: str = '-z',
    ):

        self.G1 = grasp_transfer
        self.G2 = grasp_receiver
        self.obj_pose = handover_object_pose

        self.pre_align_distance = pre_align_distance
        self.pre_align_axis = pre_align_axis
        self.retreat_distance = retreat_distance

        self.approach_steps = approach_steps
        self.ramp_steps = ramp_steps

        self.transfer_retreat_axis = transfer_retreat_axis

    def ee_pose_from_object(self, obj_pose: StampedPose, grasp: StampedPose) -> StampedPose:
        """
        T_ee = T_obj * inverse(G)
        """
        T_obj = obj_pose.to_matrix()
        G = grasp.to_matrix()

        T_ee = T_obj @ np.linalg.inv(G)
        return StampedPose.from_matrix(T_ee, obj_pose.frame_id)

    def plan_pre_alignment(self) -> Tuple[StampedPose, StampedPose]:

        # Compute receiver's final EE pose at the object, then offset along the resolved axis
        receiver_final = self.ee_pose_from_object(self.obj_pose, self.G2)

        pre_align_vec = resolve_axis(
            self.pre_align_axis,
            receiver_final.quaternion
        )

        receiver_pre = StampedPose(
            position=receiver_final.position + pre_align_vec * self.pre_align_distance,
            quaternion=receiver_final.quaternion,
            frame_id=receiver_final.frame_id
        )

        transfer_hold = self.ee_pose_from_object(self.obj_pose, self.G1)

        return receiver_pre, transfer_hold

    def plan_receiver_approach(self, receiver_pre: StampedPose) -> Tuple[List[StampedPose], StampedPose]:

        receiver_final = self.ee_pose_from_object(
            self.obj_pose,
            self.G2
        )

        receiver_traj = interpolate_pose(
            receiver_pre,
            receiver_final,
            self.approach_steps
        )

        return receiver_traj, receiver_final

    def plan_force_ramp(self, receiver_final: StampedPose) -> Tuple[List[StampedPose], List[StampedPose], List[Dict[str, float]]]:

        transfer_pose = self.ee_pose_from_object(
            self.obj_pose,
            self.G1
        )

        ramp = sigmoid_ramp(self.ramp_steps)

        forces = [
            {
                "F_transfer": 1 - a,
                "F_receiver": 0.2 + 0.8 * a
            }
            for a in ramp
        ]

        transfer_traj = [transfer_pose] * self.ramp_steps
        receiver_traj = [receiver_final] * self.ramp_steps

        return transfer_traj, receiver_traj, forces

    def plan_retreat(self) -> List[StampedPose]:

        transfer_pose = self.ee_pose_from_object(
            self.obj_pose,
            self.G1
        )

        retreat_vec = resolve_axis(
            self.transfer_retreat_axis,
            transfer_pose.quaternion
        )

        retreat_pos = transfer_pose.position + \
                      retreat_vec * self.retreat_distance

        retreat_pose = StampedPose(
            position=retreat_pos,
            quaternion=transfer_pose.quaternion,
            frame_id=transfer_pose.frame_id
        )

        return interpolate_pose(
            transfer_pose,
            retreat_pose,
            30
        )

    # ---------------------------------

    def generate_trajectory(self) -> Dict[str, Any]:

        receiver_pre, transfer_hold = self.plan_pre_alignment()

        receiver_approach, receiver_final = \
            self.plan_receiver_approach(receiver_pre)

        transfer_ramp, receiver_ramp, forces = \
            self.plan_force_ramp(receiver_final)

        retreat_traj = self.plan_retreat()

        return {
            "receiver_pre": [receiver_pre],
            "transfer_hold": [transfer_hold],
            "receiver_approach": receiver_approach,
            "force_ramp_transfer": transfer_ramp,
            "force_ramp_receiver": receiver_ramp,
            "force_schedule": forces,
            "transfer_retreat": retreat_traj
        }


class ObjectGraspPrimitive:

    def __init__(
        self,
        object_pose: StampedPose,
        grasp_transform: StampedPose,
        pregrasp_distance: float = 0.10,
        approach_axis: str = '-z',
        approach_steps: int = 50,
        grasp_steps: int = 25
    ):

        self.obj_pose = object_pose
        self.G = grasp_transform

        self.pregrasp_distance = pregrasp_distance
        self.approach_axis = approach_axis
        self.approach_steps = approach_steps
        self.grasp_steps = grasp_steps

    # ---------------------------------

    def ee_from_object(self, obj_pose: StampedPose, grasp: StampedPose) -> StampedPose:

        T_obj = obj_pose.to_matrix()
        G = grasp.to_matrix()

        T_ee = T_obj @ np.linalg.inv(G)
        return StampedPose.from_matrix(T_ee, obj_pose.frame_id)

    # ---------------------------------

    def compute_grasp_pose(self) -> StampedPose:
        return self.ee_from_object(self.obj_pose, self.G)

    # ---------------------------------

    def compute_pregrasp_pose(self, grasp_pose: StampedPose) -> StampedPose:

        approach_vec = resolve_axis(
            self.approach_axis,
            grasp_pose.quaternion
        )

        pre_pos = grasp_pose.position + \
                  approach_vec * self.pregrasp_distance

        return StampedPose(
            position=pre_pos,
            quaternion=grasp_pose.quaternion,
            frame_id=grasp_pose.frame_id
        )

    # ---------------------------------

    def generate_trajectory(self) -> Dict[str, Any]:

        grasp_pose = self.compute_grasp_pose()
        pregrasp_pose = self.compute_pregrasp_pose(grasp_pose)

        approach_traj = interpolate_pose(
            pregrasp_pose,
            grasp_pose,
            self.approach_steps
        )

        return {
            "pregrasp": pregrasp_pose,
            "approach": approach_traj,
            "grasp_pose": grasp_pose
        }


class ReleaseRetreatPrimitive:

    def __init__(
        self,
        retreat_distance: float = 0.05,
        retreat_axis: str = '-z',
        retreat_steps: int = 30,
        settle_time: float = 0.1
    ):

        self.retreat_distance = retreat_distance
        self.retreat_axis = retreat_axis
        self.retreat_steps = retreat_steps
        self.settle_time = settle_time

    def compute_retreat_pose(self, release_pose: StampedPose) -> StampedPose:

        retreat_vec = resolve_axis(
            self.retreat_axis,
            release_pose.quaternion
        )

        retreat_pos = release_pose.position + \
                      retreat_vec * self.retreat_distance

        return StampedPose(
            position=retreat_pos,
            quaternion=release_pose.quaternion,
            frame_id=release_pose.frame_id
        )


    def generate_trajectory(
        self,
        release_pose: StampedPose
    ) -> Dict[str, Any]:

        retreat_pose = self.compute_retreat_pose(release_pose)

        retreat_traj = interpolate_pose(
            release_pose,
            retreat_pose,
            self.retreat_steps
        )

        return {
            "release_pose": release_pose,
            "settle_time": self.settle_time,
            "retreat": retreat_traj
        }
