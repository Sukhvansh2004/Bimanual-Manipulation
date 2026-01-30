"""
Suction gripper grasp pose generator for a thin plate.

The plate dimensions are: 10cm x 20cm x 2cm (width x length x thickness)
Frame is attached to the center of the plate.
Suction gripper can only grab from the top and bottom (not from the sides).
"""

import numpy as np
from typing import List
from dataclasses import dataclass
from robosuite_franka_manipulation.planning.grasp_generator import GraspPose


class SuctionGraspGenerator:
    """
    Generates grasp poses for a suction gripper on a thin plate.
    
    Plate dimensions: 10cm x 20cm x 2cm (width x length x thickness)
    Frame is at the center (centroid) of the plate at (0, 0, 0).
    All generated grasp poses are in the plate frame (centroid frame).
    Suction gripper can only grasp from top and bottom surfaces.
    """
    
    def __init__(
        self,
        plate_width: float = 10.0,  # cm
        plate_length: float = 20.0,  # cm
        plate_thickness: float = 2.0,  # cm
        suction_cup_radius: float = 1.0,  # cm - radius of suction cup
        min_distance_from_edge: float = 1.5,  # cm - minimum distance from plate edge
    ):
        """
        Initialize the suction grasp generator.
        
        Args:
            plate_width: Width of the plate in cm
            plate_length: Length of the plate in cm
            plate_thickness: Thickness of the plate in cm
            suction_cup_radius: Radius of the suction cup in cm
            min_distance_from_edge: Minimum distance from plate edge for suction placement
        """
        self.plate_width = plate_width
        self.plate_length = plate_length
        self.plate_thickness = plate_thickness
        self.suction_cup_radius = suction_cup_radius
        self.min_distance_from_edge = min_distance_from_edge
        
        # Half dimensions (frame is at center)
        self.half_width = plate_width / 2.0
        self.half_length = plate_length / 2.0
        self.half_thickness = plate_thickness / 2.0
        
        # Valid suction area (excluding edges)
        self.valid_width = plate_width - 2 * min_distance_from_edge
        self.valid_length = plate_length - 2 * min_distance_from_edge
    
    def _rotation_matrix_from_axes(
        self, x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray
    ) -> np.ndarray:
        """Create a rotation matrix from three orthogonal axes."""
        R = np.column_stack([x_axis, y_axis, z_axis])
        # Ensure orthonormality
        R, _ = np.linalg.qr(R)
        return R
    
    def generate_top_grasps(
        self,
        num_grasps_x: int = 5,
        num_grasps_y: int = 5,
    ) -> List[GraspPose]:
        """
        Generate suction grasps on the top surface of the plate.
        
        Args:
            num_grasps_x: Number of grasps along the width (x-direction)
            num_grasps_y: Number of grasps along the length (y-direction)
        
        Returns:
            List of GraspPose objects (all in plate frame)
        """
        grasps = []
        
        # Generate grid of positions on top surface
        if num_grasps_x > 1:
            x_step = self.valid_width / (num_grasps_x - 1)
            x_start = -self.valid_width / 2.0
        else:
            x_step = 0
            x_start = 0.0
        
        if num_grasps_y > 1:
            y_step = self.valid_length / (num_grasps_y - 1)
            y_start = -self.valid_length / 2.0
        else:
            y_step = 0
            y_start = 0.0
        
        # Top surface is at z = half_thickness
        z_position = self.half_thickness
        
        for i in range(num_grasps_x):
            for j in range(num_grasps_y):
                x = x_start + i * x_step
                y = y_start + j * y_step
                position = np.array([x, y, z_position])
                
                # Approach from top (negative z direction)
                approach_direction = np.array([0.0, 0.0, -1.0])
                
                # Orientation: z-axis points down (approach direction), 
                # x and y axes aligned with plate x and y
                z_axis = -approach_direction  # Points down
                x_axis = np.array([1.0, 0.0, 0.0])  # Along plate width
                y_axis = np.array([0.0, 1.0, 0.0])  # Along plate length
                
                orientation = self._rotation_matrix_from_axes(x_axis, y_axis, z_axis)
                
                grasp = GraspPose(
                    position=position,
                    orientation=orientation,
                    approach_direction=approach_direction,
                    gripper_opening=0.0,  # Not applicable for suction gripper
                )
                grasps.append(grasp)
        
        return grasps
    
    def generate_bottom_grasps(
        self,
        num_grasps_x: int = 5,
        num_grasps_y: int = 5,
    ) -> List[GraspPose]:
        """
        Generate suction grasps on the bottom surface of the plate.
        
        Args:
            num_grasps_x: Number of grasps along the width (x-direction)
            num_grasps_y: Number of grasps along the length (y-direction)
        
        Returns:
            List of GraspPose objects (all in plate frame)
        """
        grasps = []
        
        # Generate grid of positions on bottom surface
        if num_grasps_x > 1:
            x_step = self.valid_width / (num_grasps_x - 1)
            x_start = -self.valid_width / 2.0
        else:
            x_step = 0
            x_start = 0.0
        
        if num_grasps_y > 1:
            y_step = self.valid_length / (num_grasps_y - 1)
            y_start = -self.valid_length / 2.0
        else:
            y_step = 0
            y_start = 0.0
        
        # Bottom surface is at z = -half_thickness
        z_position = -self.half_thickness
        
        for i in range(num_grasps_x):
            for j in range(num_grasps_y):
                x = x_start + i * x_step
                y = y_start + j * y_step
                position = np.array([x, y, z_position])
                
                # Approach from bottom (positive z direction)
                approach_direction = np.array([0.0, 0.0, 1.0])
                
                # Orientation: z-axis points up (approach direction),
                # x and y axes aligned with plate x and y (but flipped)
                z_axis = approach_direction  # Points up
                x_axis = np.array([1.0, 0.0, 0.0])  # Along plate width
                y_axis = np.array([0.0, -1.0, 0.0])  # Flipped to maintain right-handed frame
                
                orientation = self._rotation_matrix_from_axes(x_axis, y_axis, z_axis)
                
                grasp = GraspPose(
                    position=position,
                    orientation=orientation,
                    approach_direction=approach_direction,
                    gripper_opening=0.0,  # Not applicable for suction gripper
                )
                grasps.append(grasp)
        
        return grasps
    
    def generate_all_grasps(
        self,
        num_grasps_x: int = 5,
        num_grasps_y: int = 5,
        include_top: bool = True,
        include_bottom: bool = True,
    ) -> List[GraspPose]:
        """
        Generate all possible suction grasp poses on top and/or bottom surfaces.
        
        All poses are in the plate frame (centroid at origin).
        
        Args:
            num_grasps_x: Number of grasps along the width (x-direction)
            num_grasps_y: Number of grasps along the length (y-direction)
            include_top: If True, generate grasps on top surface
            include_bottom: If True, generate grasps on bottom surface
        
        Returns:
            List of all GraspPose objects (all in plate frame)
        """
        all_grasps = []
        
        if include_top:
            top_grasps = self.generate_top_grasps(num_grasps_x, num_grasps_y)
            all_grasps.extend(top_grasps)
        
        if include_bottom:
            bottom_grasps = self.generate_bottom_grasps(num_grasps_x, num_grasps_y)
            all_grasps.extend(bottom_grasps)
        
        return all_grasps
