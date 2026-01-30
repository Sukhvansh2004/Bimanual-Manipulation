"""
Collision detection utilities for grippers and plate.

Suction gripper: 4cm x 4cm x 4cm cuboid
Parallel jaw gripper: thinner cuboidal shape
"""

import numpy as np
from typing import List, Tuple, Optional
from robosuite_franka_manipulation.planning.grasp_generator import GraspPose


class GripperGeometry:
    """Represents the geometry of a gripper."""
    
    def __init__(
        self,
        width: float,
        length: float,
        height: float,
        center_offset: np.ndarray = None,
    ):
        """
        Initialize gripper geometry.
        
        Args:
            width: Width of gripper cuboid (cm)
            length: Length of gripper cuboid (cm)
            height: Height of gripper cuboid (cm)
            center_offset: Offset from grasp position to gripper center (3D vector)
        """
        self.width = width
        self.length = length
        self.height = height
        self.center_offset = center_offset if center_offset is not None else np.array([0.0, 0.0, 0.0])
        
        # Half dimensions
        self.half_width = width / 2.0
        self.half_length = length / 2.0
        self.half_height = height / 2.0
    
    def get_cuboid_vertices(self, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """
        Get the 8 vertices of the gripper cuboid in world frame.
        
        Args:
            position: Gripper position (grasp position)
            orientation: 3x3 rotation matrix
        
        Returns:
            8x3 array of vertices
        """
        # Gripper center in world frame
        gripper_center = position + orientation @ self.center_offset
        
        # Local vertices in gripper frame (centered at origin)
        local_vertices = np.array([
            [-self.half_width, -self.half_length, -self.half_height],
            [self.half_width, -self.half_length, -self.half_height],
            [self.half_width, self.half_length, -self.half_height],
            [-self.half_width, self.half_length, -self.half_height],
            [-self.half_width, -self.half_length, self.half_height],
            [self.half_width, -self.half_length, self.half_height],
            [self.half_width, self.half_length, self.half_height],
            [-self.half_width, self.half_length, self.half_height],
        ])
        
        # Transform to world frame
        world_vertices = gripper_center + (orientation @ local_vertices.T).T
        
        return world_vertices
    
    def get_bounding_box(self, position: np.ndarray, orientation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get axis-aligned bounding box (AABB) of the gripper.
        
        Args:
            position: Gripper position
            orientation: 3x3 rotation matrix
        
        Returns:
            (min_corner, max_corner) - two 3D points defining the AABB
        """
        vertices = self.get_cuboid_vertices(position, orientation)
        min_corner = np.min(vertices, axis=0)
        max_corner = np.max(vertices, axis=0)
        return min_corner, max_corner


# Define gripper geometries
SUCTION_GEOMETRY = GripperGeometry(
    width=4.0,
    length=4.0,
    height=4.0,
    center_offset=np.array([0.0, 0.0, 3.0]),  # Center is 2cm above contact point
)

# Parallel jaw gripper: thinner cuboid
# Approximate dimensions based on typical parallel jaw grippers
PARALLEL_JAW_GEOMETRY = GripperGeometry(
    width=3.0,   # Width of gripper body
    length=5.0,  # Length along approach direction
    height=2.0,  # Height (opening direction)
    center_offset=np.array([0.0, 0.0, -1.5]),  # Center is behind the jaws
)


class PlateGeometry:
    """Represents the plate geometry."""
    
    def __init__(
        self,
        width: float,
        length: float,
        thickness: float,
    ):
        """
        Initialize plate geometry.
        
        Args:
            width: Width of plate (cm)
            length: Length of plate (cm)
            thickness: Thickness of plate (cm)
        """
        self.width = width
        self.length = length
        self.thickness = thickness
        
        self.half_width = width / 2.0
        self.half_length = length / 2.0
        self.half_thickness = thickness / 2.0
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get axis-aligned bounding box of the plate (in plate frame).
        
        Returns:
            (min_corner, max_corner) - two 3D points defining the AABB
        """
        min_corner = np.array([
            -self.half_width,
            -self.half_length,
            -self.half_thickness,
        ])
        max_corner = np.array([
            self.half_width,
            self.half_length,
            self.half_thickness,
        ])
        return min_corner, max_corner
    
    def contains_point(self, point: np.ndarray) -> bool:
        """
        Check if a point is inside the plate.
        
        Args:
            point: 3D point in plate frame
        
        Returns:
            True if point is inside plate
        """
        return (
            abs(point[0]) <= self.half_width and
            abs(point[1]) <= self.half_length and
            abs(point[2]) <= self.half_thickness
        )


def aabb_overlap(
    min1: np.ndarray,
    max1: np.ndarray,
    min2: np.ndarray,
    max2: np.ndarray,
) -> bool:
    """
    Check if two axis-aligned bounding boxes overlap.
    
    Args:
        min1, max1: Bounding box 1 (min and max corners)
        min2, max2: Bounding box 2 (min and max corners)
    
    Returns:
        True if boxes overlap
    """
    return (
        min1[0] <= max2[0] and max1[0] >= min2[0] and
        min1[1] <= max2[1] and max1[1] >= min2[1] and
        min1[2] <= max2[2] and max1[2] >= min2[2]
    )


def check_gripper_plate_collision(
    gripper_geometry: GripperGeometry,
    grasp: GraspPose,
    plate: PlateGeometry,
    margin: float = 0.1,
) -> bool:
    """
    Check if a gripper collides with the plate.
    
    Args:
        gripper_geometry: Geometry of the gripper
        grasp: Grasp pose
        plate: Plate geometry
        margin: Safety margin (cm)
    
    Returns:
        True if collision detected
    """
    # Get gripper bounding box
    gripper_min, gripper_max = gripper_geometry.get_bounding_box(grasp.position, grasp.orientation)
    
    # Get plate bounding box
    plate_min, plate_max = plate.get_bounding_box()
    
    # Add margin to plate
    plate_min = plate_min - margin
    plate_max = plate_max + margin
    
    # Check overlap
    return aabb_overlap(gripper_min, gripper_max, plate_min, plate_max)


def check_gripper_gripper_collision(
    gripper1_geometry: GripperGeometry,
    grasp1: GraspPose,
    gripper2_geometry: GripperGeometry,
    grasp2: GraspPose,
    margin: float = 0.1,
) -> bool:
    """
    Check if two grippers collide with each other.
    
    Args:
        gripper1_geometry: Geometry of first gripper
        grasp1: First grasp pose
        gripper2_geometry: Geometry of second gripper
        grasp2: Second grasp pose
        margin: Safety margin (cm)
    
    Returns:
        True if collision detected
    """
    # Get bounding boxes
    min1, max1 = gripper1_geometry.get_bounding_box(grasp1.position, grasp1.orientation)
    min2, max2 = gripper2_geometry.get_bounding_box(grasp2.position, grasp2.orientation)
    
    # Add margin
    min1 = min1 - margin
    max1 = max1 + margin
    min2 = min2 - margin
    max2 = max2 + margin
    
    # Check overlap
    return aabb_overlap(min1, max1, min2, max2)


def filter_grasps_no_plate_collision(
    grasps: List[GraspPose],
    gripper_geometry: GripperGeometry,
    plate: PlateGeometry,
    margin: float = 0.1,
) -> List[GraspPose]:
    """
    Filter grasps that don't collide with the plate.
    
    Args:
        grasps: List of grasp poses
        gripper_geometry: Geometry of the gripper
        plate: Plate geometry
        margin: Safety margin (cm)
    
    Returns:
        Filtered list of grasps without plate collision
    """
    filtered = []
    for grasp in grasps:
        if not check_gripper_plate_collision(gripper_geometry, grasp, plate, margin):
            filtered.append(grasp)
    return filtered


def find_feasible_grasp_pairs(
    suction_grasps: List[GraspPose],
    parallel_jaw_grasps: List[GraspPose],
    plate_width: float = 10.0,
    plate_length: float = 20.0,
    plate_thickness: float = 2.0,
    margin: float = 0.1,
) -> List[Tuple[GraspPose, GraspPose]]:
    """
    Find all feasible pairs of grasps (suction, parallel_jaw) that:
    1. Don't collide with the plate
    2. Don't collide with each other
    
    Args:
        suction_grasps: List of suction grasp poses
        parallel_jaw_grasps: List of parallel jaw grasp poses
        plate_width: Width of plate (cm)
        plate_length: Length of plate (cm)
        plate_thickness: Thickness of plate (cm)
        margin: Safety margin for collision detection (cm)
    
    Returns:
        List of feasible (suction_grasp, parallel_jaw_grasp) pairs
    """
    # Create plate geometry
    plate = PlateGeometry(plate_width, plate_length, plate_thickness)
    
    # Filter grasps that don't collide with plate
    print(f"Filtering {len(suction_grasps)} suction grasps...")
    feasible_suction = filter_grasps_no_plate_collision(
        suction_grasps, SUCTION_GEOMETRY, plate, margin
    )
    print(f"  -> {len(feasible_suction)} feasible suction grasps")
    
    print(f"Filtering {len(parallel_jaw_grasps)} parallel jaw grasps...")
    feasible_parallel_jaw = filter_grasps_no_plate_collision(
        parallel_jaw_grasps, PARALLEL_JAW_GEOMETRY, plate, margin
    )
    print(f"  -> {len(feasible_parallel_jaw)} feasible parallel jaw grasps")
    
    # Create all pairs and check gripper-gripper collision
    print(f"\nChecking {len(feasible_suction)} x {len(feasible_parallel_jaw)} = {len(feasible_suction) * len(feasible_parallel_jaw)} pairs...")
    feasible_pairs = []
    
    for i, suction_grasp in enumerate(feasible_suction):
        for parallel_jaw_grasp in feasible_parallel_jaw:
            # Check if grippers collide with each other
            if not check_gripper_gripper_collision(
                SUCTION_GEOMETRY, suction_grasp,
                PARALLEL_JAW_GEOMETRY, parallel_jaw_grasp,
                margin
            ):
                feasible_pairs.append((suction_grasp, parallel_jaw_grasp))
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(feasible_suction)} suction grasps, found {len(feasible_pairs)} feasible pairs so far...")
    
    print(f"\nFound {len(feasible_pairs)} feasible pairs")
    
    return feasible_pairs


def find_infeasible_grasp_pairs(
    suction_grasps: List[GraspPose],
    parallel_jaw_grasps: List[GraspPose],
    plate_width: float = 10.0,
    plate_length: float = 20.0,
    plate_thickness: float = 2.0,
    margin: float = 0.1,
) -> List[Tuple[GraspPose, GraspPose, str]]:
    """
    Find all infeasible pairs of grasps (suction, parallel_jaw) that:
    1. Collide with the plate, OR
    2. Collide with each other
    
    Args:
        suction_grasps: List of suction grasp poses
        parallel_jaw_grasps: List of parallel jaw grasp poses
        plate_width: Width of plate (cm)
        plate_length: Length of plate (cm)
        plate_thickness: Thickness of plate (cm)
        margin: Safety margin for collision detection (cm)
    
    Returns:
        List of (suction_grasp, parallel_jaw_grasp, reason) tuples where reason is:
        - "plate_collision_suction": suction gripper collides with plate
        - "plate_collision_parallel_jaw": parallel jaw gripper collides with plate
        - "gripper_collision": grippers collide with each other
    """
    plate = PlateGeometry(plate_width, plate_length, plate_thickness)
    infeasible_pairs = []
    
    print(f"\nFinding infeasible pairs...")
    total_pairs = len(suction_grasps) * len(parallel_jaw_grasps)
    processed = 0
    
    for suction_grasp in suction_grasps:
        for parallel_jaw_grasp in parallel_jaw_grasps:
            processed += 1
            reasons = []
            
            # Check suction gripper collision with plate
            if check_gripper_plate_collision(SUCTION_GEOMETRY, suction_grasp, plate, margin):
                reasons.append("plate_collision_suction")
            
            # Check parallel jaw gripper collision with plate
            if check_gripper_plate_collision(PARALLEL_JAW_GEOMETRY, parallel_jaw_grasp, plate, margin):
                reasons.append("plate_collision_parallel_jaw")
            
            # Check gripper-gripper collision (only if both don't collide with plate)
            if not reasons:
                if check_gripper_gripper_collision(
                    SUCTION_GEOMETRY, suction_grasp,
                    PARALLEL_JAW_GEOMETRY, parallel_jaw_grasp,
                    margin
                ):
                    reasons.append("gripper_collision")
            
            # If any collision detected, add to infeasible
            if reasons:
                # Use the first reason as primary
                infeasible_pairs.append((suction_grasp, parallel_jaw_grasp, reasons[0]))
        
        if processed % 100 == 0:
            print(f"  Processed {processed}/{total_pairs} pairs, found {len(infeasible_pairs)} infeasible so far...")
    
    print(f"Found {len(infeasible_pairs)} infeasible pairs")
    
    return infeasible_pairs
