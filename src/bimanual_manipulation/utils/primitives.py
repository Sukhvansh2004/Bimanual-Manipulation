# TODO: Switch from linear translation → minimum jerk trajectories.

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Union, Callable, Any, Tuple
from scipy.spatial.transform import Rotation as R, Slerp

from bimanual_manipulation.transforms import StampedPose, Transform
from bimanual_manipulation import constants

@dataclass
class HandoverTrajectory:
    """Typed container for the output of :meth:`DualArmHandoverPlanner.generate_trajectory`.

    All poses are in world frame.
    """

    receiver_pre: List[StampedPose] = field(default_factory=list)
    transfer_hold: List[StampedPose] = field(default_factory=list)
    receiver_approach: List[StampedPose] = field(default_factory=list)
    force_ramp_transfer: List[StampedPose] = field(default_factory=list)
    force_ramp_receiver: List[StampedPose] = field(default_factory=list)
    force_schedule: List[Dict[str, float]] = field(default_factory=list)
    transfer_retreat: List[StampedPose] = field(default_factory=list)


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


def compute_ee_pose(obj_pose: StampedPose, grasp: StampedPose) -> StampedPose:
    """
    Compute end-effector pose in world frame from an object pose and a grasp transform.

    Uses Transform composition: T_ee = T_obj @ inv(G)
    where G represents the object pose in the EE frame (ee <- object).

    Args:
        obj_pose: Object pose in world frame (world <- object)
        grasp: Grasp transform — object pose relative to EE (ee <- object)

    Returns:
        EE pose in the same frame as obj_pose (world <- ee)
    """
    T_obj = Transform(
        translation=obj_pose.position,
        rotation=obj_pose.quaternion,
        parent_frame=obj_pose.frame_id,
        child_frame="object",
    )
    G = Transform(
        translation=grasp.position,
        rotation=grasp.quaternion,
        parent_frame="ee",
        child_frame="object",
    )
    # world <- object  @  object <- ee  =  world <- ee
    T_ee = T_obj.compose(G.inverse())
    return StampedPose(
        position=T_ee.translation,
        quaternion=T_ee.rotation,
        frame_id=T_ee.parent_frame,
    )


def extract_grasp_transform(ee_pose: StampedPose, obj_pose: StampedPose) -> StampedPose:
    """
    Extract the grasp transform from an EE pose and an object pose.

    Computes G = inv(T_ee) @ T_obj, i.e. the object pose in the EE frame.

    Args:
        ee_pose: End-effector pose in world frame (world <- ee)
        obj_pose: Object pose in world frame (world <- object)

    Returns:
        Grasp as StampedPose representing object-in-EE (ee <- object)
    """
    T_ee = Transform(
        translation=ee_pose.position,
        rotation=ee_pose.quaternion,
        parent_frame=ee_pose.frame_id,
        child_frame="ee",
    )
    T_obj = Transform(
        translation=obj_pose.position,
        rotation=obj_pose.quaternion,
        parent_frame=obj_pose.frame_id,
        child_frame="object",
    )
    # ee <- world  @  world <- object  =  ee <- object
    G = T_ee.inverse().compose(T_obj)
    return StampedPose(
        position=G.translation,
        quaternion=G.rotation,
        frame_id=G.parent_frame,
    )


def sigmoid_ramp(steps: int) -> np.ndarray:
    x = np.linspace(-6, 6, steps)
    s = 1 / (1 + np.exp(-x))
    return (s - s.min()) / (s.max() - s.min())


class DualArmHandoverPlanner:
    """
    Plans a dual-arm handover sequence between a transfer arm and a receiver arm.

    All poses are expected in the world frame. Grasp transforms (G1, G2) represent
    the object pose in each EE's frame (ee <- object).

    The planner produces five phases:
        1. Pre-alignment  — receiver moves to a standoff pose offset from the object.
        2. Approach        — receiver interpolates from standoff to the grasp pose.
        3. Force ramp      — both arms hold; grip force transfers via sigmoid ramp.
        4. Retreat          — transfer arm backs away along its retreat axis.

    Args:
        grasp_transfer: Grasp transform for the transfer arm (ee <- object).
        grasp_receiver: Grasp transform for the receiver arm (ee <- object).
        handover_object_pose: Object pose in world frame at the handover location.
        pre_align_distance: Standoff distance along pre_align_axis (metres).
        pre_align_axis: EE-frame axis for pre-alignment offset ('x','y','z','-x','-y','-z').
        retreat_distance: Distance the transfer arm retreats after release (metres).
        approach_steps: Number of interpolation steps for receiver approach.
        ramp_steps: Number of steps for the force transfer sigmoid ramp.
        transfer_retreat_axis: EE-frame axis for transfer arm retreat.
    """

    def __init__(
        self,
        grasp_transfer: StampedPose,
        grasp_receiver: StampedPose,
        handover_object_pose: StampedPose,
        pre_align_distance: float = constants.HANDOVER_PRE_ALIGN_DISTANCE,
        pre_align_axis: str = constants.HANDOVER_PRE_ALIGN_AXIS,
        retreat_distance: float = constants.HANDOVER_RETREAT_DISTANCE,
        approach_steps: int = constants.HANDOVER_APPROACH_STEPS,
        ramp_steps: int = constants.HANDOVER_RAMP_STEPS,
        transfer_retreat_axis: str = constants.HANDOVER_RETREAT_AXIS,
        retreat_steps: int = constants.HANDOVER_RETREAT_STEPS,
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
        self.retreat_steps = retreat_steps

    def plan_pre_alignment(self) -> Tuple[StampedPose, StampedPose]:
        """
        Compute the receiver's standoff pose and the transfer arm's hold pose.

        The standoff pose is offset from the receiver's final grasp pose along
        ``pre_align_axis`` in the receiver-EE frame.

        Returns:
            (receiver_pre, transfer_hold) — both in world frame.
        """

        # Compute receiver's final EE pose at the object, then offset along the resolved axis
        receiver_final = compute_ee_pose(self.obj_pose, self.G2)

        pre_align_vec = resolve_axis(
            self.pre_align_axis,
            receiver_final.quaternion
        )

        receiver_pre = StampedPose(
            position=receiver_final.position + pre_align_vec * self.pre_align_distance,
            quaternion=receiver_final.quaternion,
            frame_id=receiver_final.frame_id
        )

        transfer_hold = compute_ee_pose(self.obj_pose, self.G1)

        return receiver_pre, transfer_hold

    def plan_receiver_approach(self, receiver_pre: StampedPose) -> Tuple[List[StampedPose], StampedPose]:
        """
        Interpolate the receiver from its standoff pose to the final grasp pose.

        Args:
            receiver_pre: Receiver standoff pose in world frame.

        Returns:
            (trajectory, receiver_final) — list of interpolated world-frame poses
            and the final grasp pose.
        """

        receiver_final = compute_ee_pose(
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
        """
        Generate static-pose trajectories for both arms during force transfer.

        Both arms hold their grasp poses while grip forces transition from
        transfer to receiver via a sigmoid ramp.

        Args:
            receiver_final: Receiver's final grasp pose in world frame.

        Returns:
            (transfer_traj, receiver_traj, force_schedule) — pose lists and
            per-step force weights {"F_transfer", "F_receiver"}.
        """

        transfer_pose = compute_ee_pose(
            self.obj_pose,
            self.G1
        )

        if self.ramp_steps <= 0:
            return [], [], []

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
        """
        Generate a retreat trajectory for the transfer arm after release.

        The arm moves along ``transfer_retreat_axis`` (resolved in its EE frame)
        by ``retreat_distance`` metres.

        Returns:
            List of world-frame poses from the hold pose to the retreated pose.
        """

        transfer_pose = compute_ee_pose(
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
            self.retreat_steps,
        )

    # ---------------------------------

    def generate_trajectory(self) -> HandoverTrajectory:
        """
        Generate the full handover trajectory.

        All output poses are in world frame.

        Returns:
            :class:`HandoverTrajectory` with per-phase pose lists.
        """

        receiver_pre, transfer_hold = self.plan_pre_alignment()

        receiver_approach, receiver_final = \
            self.plan_receiver_approach(receiver_pre)

        transfer_ramp, receiver_ramp, forces = \
            self.plan_force_ramp(receiver_final)

        retreat_traj = self.plan_retreat()

        return HandoverTrajectory(
            receiver_pre=[receiver_pre],
            transfer_hold=[transfer_hold],
            receiver_approach=receiver_approach,
            force_ramp_transfer=transfer_ramp,
            force_ramp_receiver=receiver_ramp,
            force_schedule=forces,
            transfer_retreat=retreat_traj,
        )


class ObjectGraspPrimitive:
    """
    Plans an approach-and-grasp trajectory for a single arm.

    All poses are expected in the world frame. The grasp transform (G)
    represents the object pose in the EE's frame (ee <- object).

    The planner produces:
        1. Grasp pose     — EE pose that places the gripper at the object.
        2. Pre-grasp pose — offset from grasp along ``approach_axis`` in EE frame.
        3. Approach trajectory — interpolation from pre-grasp to grasp.

    Args:
        object_pose: Object pose in world frame (world <- object).
        grasp_transform: Grasp transform — object in EE frame (ee <- object).
        pregrasp_distance: Standoff distance along approach_axis (metres).
        approach_axis: EE-frame axis for the approach direction.
        approach_steps: Number of interpolation steps.
        grasp_steps: Number of steps to hold at grasp (unused currently).
    """

    def __init__(
        self,
        object_pose: StampedPose,
        grasp_transform: StampedPose,
        pregrasp_distance: float = constants.HANDOVER_PRE_ALIGN_DISTANCE,
        approach_axis: str = constants.HANDOVER_PRE_ALIGN_AXIS,
        approach_steps: int = constants.HANDOVER_APPROACH_STEPS,
        grasp_steps: int = constants.HANDOVER_GRASP_STEPS
    ):

        self.obj_pose = object_pose
        self.G = grasp_transform

        self.pregrasp_distance = pregrasp_distance
        self.approach_axis = approach_axis
        self.approach_steps = approach_steps
        self.grasp_steps = grasp_steps

    # ---------------------------------

    def compute_pregrasp_pose(self, grasp_pose: StampedPose) -> StampedPose:
        """
        Compute the pre-grasp standoff pose.

        Offsets from ``grasp_pose`` along ``approach_axis`` (resolved in the
        grasp pose's EE frame) by ``pregrasp_distance``.

        Args:
            grasp_pose: The target grasp pose in world frame.

        Returns:
            Pre-grasp pose in world frame.
        """

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
        """
        Generate the full grasp trajectory dictionary.

        All output poses are in world frame.

        Returns:
            Dict with keys:
                - pregrasp: StampedPose — standoff pose.
                - approach: List[StampedPose] — approach trajectory.
                - grasp_pose: StampedPose — final grasp pose.
        """

        grasp_pose = compute_ee_pose(self.obj_pose, self.G)
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
    """
    Plans a retreat trajectory after releasing an object.

    The retreat direction is resolved in the release pose's EE frame.
    All poses are expected and returned in world frame.

    Args:
        retreat_distance: How far to retreat (metres).
        retreat_axis: EE-frame axis for the retreat direction.
        retreat_steps: Number of interpolation steps.
        settle_time: Time to wait after release before retreating (seconds).
    """

    def __init__(
        self,
        retreat_distance: float = constants.HANDOVER_RETREAT_DISTANCE,
        retreat_axis: str = constants.HANDOVER_RETREAT_AXIS,
        retreat_steps: int = constants.HANDOVER_RETREAT_STEPS,
        settle_time: float = constants.HANDOVER_SETTLE_TIME
    ):

        self.retreat_distance = retreat_distance
        self.retreat_axis = retreat_axis
        self.retreat_steps = retreat_steps
        self.settle_time = settle_time

    def compute_retreat_pose(self, release_pose: StampedPose) -> StampedPose:
        """
        Compute the retreated EE pose.

        Offsets from ``release_pose`` along ``retreat_axis`` (resolved in the
        release pose's EE frame) by ``retreat_distance``.

        Args:
            release_pose: The EE pose at the moment of release, in world frame.

        Returns:
            Retreated pose in world frame.
        """

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
        """
        Generate the retreat trajectory dictionary.

        Args:
            release_pose: The EE pose at release, in world frame.

        Returns:
            Dict with keys:
                - release_pose: StampedPose — the input release pose.
                - settle_time: float — seconds to wait before retreat.
                - retreat: List[StampedPose] — retreat trajectory in world frame.
        """

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
