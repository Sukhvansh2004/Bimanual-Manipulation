#!/usr/bin/env python3
"""
Comprehensive tests for Pydantic schema models (schema.py).

Covers DetectionSchema, GraspSchema, service request/response models,
validators, serialization, and edge cases.
"""

import pytest
import numpy as np
import time

from bimanual_manipulation import constants
from bimanual_manipulation.transforms import StampedPose
from bimanual_manipulation.schema import (
    BaseSchema,
    DetectionSchema,
    GraspSchema,
    ResponseStatus,
    FastSAMRequest,
    FastSAMResponse,
    GraspRequest,
    GraspResponse,
    MoveToPoseRequest,
    FollowTrajectoryRequest,
    GetArmStatusRequest,
    CancelMoveRequest,
    MoveToJointConfigRequest,
    FollowJointTrajectoryRequest,
    MoveToPoseResponse,
    ArmStatusResponse,
    ServiceInfo,
    RegisterServiceRequest,
    LookupServiceRequest,
    CallServiceRequest,
    ListServicesRequest,
    ServiceListResponse,
    ServiceLookupResponse,
)


# ═══════════════════════════════════════════════════════════════════════════════
# DetectionSchema
# ═══════════════════════════════════════════════════════════════════════════════

class TestDetectionSchema:
    """Test DetectionSchema creation, validation, and serialization."""

    def test_basic_creation(self):
        pose = StampedPose(pos=[1, 2, 3], quat=[0, 0, 0, 1], frame_id="world")
        det = DetectionSchema(name="object_0", id=0, pose=pose)
        assert det.name == "object_0"
        assert det.id == 0
        assert det.pcd == []
        assert det.mask is None

    def test_with_point_cloud(self):
        pose = StampedPose(pos=[0, 0, 0], quat=[0, 0, 0, 1])
        pcd = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        det = DetectionSchema(name="cup", id="cup_1", pose=pose, pcd=pcd)
        assert len(det.pcd) == 2
        assert det.pcd[0] == [1.0, 2.0, 3.0]

    def test_with_mask(self):
        pose = StampedPose(pos=[0, 0, 0], quat=[0, 0, 0, 1])
        mask = [[0, 1, 0], [1, 1, 0]]
        det = DetectionSchema(name="obj", id=0, pose=pose, mask=mask)
        assert det.mask == [[0, 1, 0], [1, 1, 0]]

    def test_numpy_pcd_conversion(self):
        """pcd validator should accept numpy arrays and convert to list."""
        pose = StampedPose(pos=[0, 0, 0], quat=[0, 0, 0, 1])
        pcd_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        det = DetectionSchema(name="obj", id=0, pose=pose, pcd=pcd_np)
        assert isinstance(det.pcd, list)
        assert len(det.pcd) == 2

    def test_numpy_mask_conversion(self):
        """mask validator should accept numpy arrays and convert to list."""
        pose = StampedPose(pos=[0, 0, 0], quat=[0, 0, 0, 1])
        mask_np = np.array([[0, 1], [1, 0]])
        det = DetectionSchema(name="obj", id=0, pose=pose, mask=mask_np)
        assert isinstance(det.mask, list)

    def test_string_id(self):
        """id field should accept both int and str."""
        pose = StampedPose(pos=[0, 0, 0], quat=[0, 0, 0, 1])
        det = DetectionSchema(name="obj", id="custom_id", pose=pose)
        assert det.id == "custom_id"

    def test_serialization_roundtrip(self):
        """Test model_dump and reconstruction."""
        pose = StampedPose(pos=[1, 2, 3], quat=[0, 0, 0, 1])
        det = DetectionSchema(
            name="obj_0", id=0, pose=pose,
            pcd=[[0.1, 0.2, 0.3]], mask=[[1, 0]]
        )
        d = det.model_dump()
        assert isinstance(d, dict)
        assert d["name"] == "obj_0"
        assert "pose" in d


# ═══════════════════════════════════════════════════════════════════════════════
# GraspSchema
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraspSchema:
    """Test GraspSchema creation and validation."""

    def test_basic_creation(self):
        grasp = GraspSchema(
            pos=[0.5, 0.1, 0.3],
            quat=[0, 0, 0, 1],
            score=0.95,
            id=0,
            gripper_type="franka_panda",
            frame_id="world",
        )
        assert grasp.score == 0.95
        assert grasp.gripper_type == "franka_panda"
        assert grasp.pcd_size == 0

    def test_inherits_stamped_pose(self):
        """GraspSchema extends StampedPose — it should have position/quaternion."""
        grasp = GraspSchema(
            pos=[1, 2, 3], quat=[0, 0, 0, 1],
            score=0.5, id=1, gripper_type="franka_panda",
        )
        np.testing.assert_array_almost_equal(grasp.position, [1, 2, 3])
        np.testing.assert_array_almost_equal(grasp.quaternion, [0, 0, 0, 1])

    def test_invalid_gripper_type_raises(self):
        """Invalid gripper_type should raise ValueError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GraspSchema(
                pos=[0, 0, 0], quat=[0, 0, 0, 1],
                score=0.9, id=0,
                gripper_type="invalid_gripper",
            )

    def test_valid_gripper_types(self):
        """Both mapped gripper types should be accepted."""
        for gtype in constants.GRIPPER_TYPE_MAPPING.values():
            grasp = GraspSchema(
                pos=[0, 0, 0], quat=[0, 0, 0, 1],
                score=0.5, id=0, gripper_type=gtype,
            )
            assert grasp.gripper_type == gtype

    def test_pcd_size_field(self):
        grasp = GraspSchema(
            pos=[0, 0, 0], quat=[0, 0, 0, 1],
            score=0.8, id=0, gripper_type="franka_panda",
            pcd_size=1024,
        )
        assert grasp.pcd_size == 1024

    def test_string_id(self):
        grasp = GraspSchema(
            pos=[0, 0, 0], quat=[0, 0, 0, 1],
            score=0.5, id="grasp_custom",
            gripper_type="franka_panda",
        )
        assert grasp.id == "grasp_custom"


# ═══════════════════════════════════════════════════════════════════════════════
# ResponseStatus
# ═══════════════════════════════════════════════════════════════════════════════

class TestResponseStatus:
    """Test ResponseStatus base model."""

    def test_defaults(self):
        resp = ResponseStatus()
        assert resp.status == "success"
        assert resp.message is None
        assert isinstance(resp.timestamp, float)

    def test_custom_status(self):
        resp = ResponseStatus(status="error", message="Something went wrong")
        assert resp.status == "error"
        assert resp.message == "Something went wrong"

    def test_timestamp_auto_set(self):
        before = time.time()
        resp = ResponseStatus()
        after = time.time()
        assert before <= resp.timestamp <= after


# ═══════════════════════════════════════════════════════════════════════════════
# FastSAM Request/Response
# ═══════════════════════════════════════════════════════════════════════════════

class TestFastSAMSchemas:
    """Test FastSAM service request/response schemas."""

    def test_request_basic(self):
        req = FastSAMRequest(command="GET_LATEST_DETECTION")
        assert req.command == "GET_LATEST_DETECTION"
        assert req.prompt is None

    def test_request_with_prompt(self):
        req = FastSAMRequest(command="SET_PROMPT", prompt="red cube")
        assert req.prompt == "red cube"

    def test_response_with_detections(self):
        pose = StampedPose(pos=[0, 0, 0], quat=[0, 0, 0, 1])
        det = DetectionSchema(name="obj", id=0, pose=pose)
        resp = FastSAMResponse(detections=[det])
        assert len(resp.detections) == 1

    def test_response_empty_detections(self):
        resp = FastSAMResponse(detections=[])
        assert resp.detections == []


# ═══════════════════════════════════════════════════════════════════════════════
# Grasp Request/Response
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraspRequestResponse:
    """Test Grasp service request/response schemas."""

    def test_request_basic(self):
        pose = StampedPose(pos=[0, 0, 0], quat=[0, 0, 0, 1])
        req = GraspRequest(
            command="GET_GRASPS",
            pcd=[[0.1, 0.2, 0.3]],
            object_pose=pose,
            num_grasps=5,
            collision_check="Inter",
            gripper_type=["finger"],
        )
        assert req.num_grasps == 5
        assert req.gripper_type == ["finger"]

    def test_request_numpy_pcd(self):
        """pcd should accept numpy arrays."""
        pose = StampedPose(pos=[0, 0, 0], quat=[0, 0, 0, 1])
        pcd_np = np.array([[1.0, 2.0, 3.0]])
        req = GraspRequest(
            command="GET_GRASPS",
            pcd=pcd_np,
            object_pose=pose,
            collision_check=None,
            gripper_type=["finger"],
        )
        assert isinstance(req.pcd, list)

    def test_request_invalid_gripper_type(self):
        from pydantic import ValidationError
        pose = StampedPose(pos=[0, 0, 0], quat=[0, 0, 0, 1])
        with pytest.raises(ValidationError):
            GraspRequest(
                command="GET_GRASPS",
                collision_check=None,
                gripper_type=["nonexistent"],
                object_pose=pose,
            )

    def test_request_empty_gripper_type(self):
        from pydantic import ValidationError
        pose = StampedPose(pos=[0, 0, 0], quat=[0, 0, 0, 1])
        with pytest.raises(ValidationError):
            GraspRequest(
                command="GET_GRASPS",
                collision_check=None,
                gripper_type=[],  # min_length=1
                object_pose=pose,
            )

    def test_response_with_grasps(self):
        grasp = GraspSchema(
            pos=[0, 0, 0], quat=[0, 0, 0, 1],
            score=0.9, id=0, gripper_type="franka_panda",
        )
        resp = GraspResponse(
            grasps=[grasp],
            num_grasps_predicted=1,
            pcd_size=512,
            gripper_type=["finger"],
        )
        assert resp.num_grasps_predicted == 1
        assert len(resp.grasps) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# MoveToPose / Joint Config Requests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMoveToPoseSchemas:
    """Test MoveToPose service schemas."""

    def test_move_request(self):
        pose = StampedPose(pos=[0.5, 0, 0.5], quat=[0, 0, 0, 1])
        req = MoveToPoseRequest(command="MOVE_TO_POSE", pose=pose)
        assert req.robot_id == 0
        assert req.arm == "right"
        assert req.gripper_action == 0.0

    def test_follow_trajectory_request(self):
        poses = [
            StampedPose(pos=[0, 0, 0], quat=[0, 0, 0, 1]),
            StampedPose(pos=[1, 0, 0], quat=[0, 0, 0, 1]),
        ]
        req = FollowTrajectoryRequest(command="FOLLOW_TRAJECTORY", trajectory=poses)
        assert len(req.trajectory) == 2
        assert req.smoothen is False

    def test_cancel_request(self):
        req = CancelMoveRequest(command="CANCEL_MOVE", robot_id=1, arm="right")
        assert req.robot_id == 1

    def test_get_status_request(self):
        req = GetArmStatusRequest(command="GET_ARM_STATUS")
        assert req.arm == "right"

    def test_joint_config_request(self):
        req = MoveToJointConfigRequest(
            command="MOVE_TO_JOINT_CONFIG",
            joint_config=[0.0] * 7,
        )
        assert len(req.joint_config) == 7

    def test_joint_config_wrong_length(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            MoveToJointConfigRequest(
                command="MOVE_TO_JOINT_CONFIG",
                joint_config=[0.0] * 5,  # Too short
            )

    def test_follow_joint_trajectory_request(self):
        req = FollowJointTrajectoryRequest(
            command="FOLLOW_JOINT_TRAJECTORY",
            joint_trajectory=[[0.0] * 7, [0.1] * 7],
        )
        assert len(req.joint_trajectory) == 2

    def test_move_response(self):
        resp = MoveToPoseResponse(status="success", message="Move complete")
        assert resp.status == "success"

    def test_arm_status_response(self):
        resp = ArmStatusResponse(status_code=-1)
        assert resp.status_code == -1


# ═══════════════════════════════════════════════════════════════════════════════
# Service Manager Schemas
# ═══════════════════════════════════════════════════════════════════════════════

class TestServiceManagerSchemas:
    """Test service manager schemas."""

    def test_service_info(self):
        info = ServiceInfo(name="test_svc", host="localhost", port=5555, type="mock")
        assert info.name == "test_svc"
        assert info.description == ""
        assert isinstance(info.registered_at, float)

    def test_register_request(self):
        req = RegisterServiceRequest(
            name="my_service", host="localhost", port=5560, type="custom"
        )
        assert req.command == "REGISTER"

    def test_lookup_request(self):
        req = LookupServiceRequest(name="fastsam_service")
        assert req.command == "LOOKUP"

    def test_call_service_request(self):
        req = CallServiceRequest(
            service_name="grasp_service",
            request={"command": "GET_GRASPS", "num_grasps": 5},
        )
        assert req.service_name == "grasp_service"

    def test_list_services_request(self):
        req = ListServicesRequest()
        assert req.command == "LIST_SERVICES"

    def test_service_list_response(self):
        info = ServiceInfo(name="svc", host="localhost", port=5555, type="mock")
        resp = ServiceListResponse(services={"svc": info})
        assert "svc" in resp.services

    def test_service_lookup_response_empty(self):
        resp = ServiceLookupResponse(service=None)
        assert resp.service is None
