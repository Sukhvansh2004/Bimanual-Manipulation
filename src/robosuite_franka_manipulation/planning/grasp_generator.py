"""
Grasp pose generator for parallel jaw gripper on a thin plate.

The plate dimensions are: 10cm x 20cm x 2cm (width x length x thickness)
Frame is attached to the center of the plate.
Gripper can only grab from the sides (not from the top).
Maximum gripper opening: 5cm
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class GraspPose:
    """Represents a grasp pose with position and orientation."""
    position: np.ndarray  # 3D position (x, y, z)
    orientation: np.ndarray  # Rotation matrix (3x3) or quaternion
    approach_direction: np.ndarray  # Direction the gripper approaches from
    gripper_opening: float  # Required gripper opening in cm
    
    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 homogeneous transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self.orientation
        T[:3, 3] = self.position
        return T


class PlateGraspGenerator:
    """
    Generates grasp poses for a parallel jaw gripper on a thin plate.
    
    Plate dimensions: 10cm x 20cm x 2cm (width x length x thickness)
    Frame is at the center (centroid) of the plate at (0, 0, 0).
    All generated grasp poses are in the plate frame (centroid frame).
    """
    
    def __init__(
        self,
        plate_width: float = 10.0,  # cm
        plate_length: float = 20.0,  # cm
        plate_thickness: float = 2.0,  # cm
        max_gripper_opening: float = 5.0,  # cm
        grasp_offset: float = 1.0,  # cm - distance from plate edge to gripper center
    ):
        """
        Initialize the grasp generator.
        
        Args:
            plate_width: Width of the plate in cm
            plate_length: Length of the plate in cm
            plate_thickness: Thickness of the plate in cm
            max_gripper_opening: Maximum opening of the gripper in cm
            grasp_offset: Distance from plate edge to gripper center in cm
        """
        self.plate_width = plate_width
        self.plate_length = plate_length
        self.plate_thickness = plate_thickness
        self.max_gripper_opening = max_gripper_opening
        self.grasp_offset = grasp_offset
        
        # Verify gripper can grasp the plate
        if self.plate_thickness > self.max_gripper_opening:
            raise ValueError(
                f"Plate thickness ({plate_thickness}cm) exceeds "
                f"max gripper opening ({max_gripper_opening}cm)"
            )
        
        # Half dimensions (frame is at center)
        self.half_width = plate_width / 2.0
        self.half_length = plate_length / 2.0
        self.half_thickness = plate_thickness / 2.0
    
    def _rotation_matrix_from_axes(
        self, x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray
    ) -> np.ndarray:
        """Create a rotation matrix from three orthogonal axes."""
        R = np.column_stack([x_axis, y_axis, z_axis])
        # Ensure orthonormality
        R, _ = np.linalg.qr(R)
        return R
    
    def generate_grasps_along_edge(
        self,
        edge_center: np.ndarray,
        edge_direction: np.ndarray,
        edge_length: float,
        approach_direction: np.ndarray,
        num_grasps: int = 5,
        gripper_axis: np.ndarray = None,
    ) -> List[GraspPose]:
        """
        Generate grasps along a single edge of the plate.
        
        Args:
            edge_center: Center point of the edge (3D)
            edge_direction: Direction along the edge (normalized)
            edge_length: Length of the edge
            approach_direction: Direction from which gripper approaches (normalized)
            num_grasps: Number of grasp poses along this edge
            gripper_axis: Direction of gripper opening (normalized). If None, computed automatically.
        
        Returns:
            List of GraspPose objects
        """
        grasps = []
        
        # Generate positions along the edge
        step = edge_length / (num_grasps + 1)
        start_offset = -edge_length / 2.0 + step
        
        for i in range(num_grasps):
            # Position along the edge
            offset = start_offset + i * step
            position = edge_center + offset * edge_direction + self.grasp_offset * approach_direction
            
            # Gripper orientation:
            # - z-axis: approach direction (gripper approaches from this direction)
            # - y-axis: gripper opening direction (perpendicular to plate surface)
            # - x-axis: along the edge (computed from cross product)
            
            z_axis = -approach_direction  # Negative because we approach from outside
            
            if gripper_axis is None:
                # Default: Gripper opens in z direction (plate thickness direction)
                # This is correct for side grasps
                y_axis = np.array([0.0, 0.0, 1.0])
            else:
                y_axis = gripper_axis
            
            # Ensure y_axis is perpendicular to z_axis
            y_axis = y_axis - np.dot(y_axis, z_axis) * z_axis
            y_axis = y_axis / np.linalg.norm(y_axis)
            
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis)
            
            # Recompute y_axis to ensure orthonormality
            y_axis = np.cross(z_axis, x_axis)
            
            orientation = self._rotation_matrix_from_axes(x_axis, y_axis, z_axis)
            
            grasp = GraspPose(
                position=position,
                orientation=orientation,
                approach_direction=approach_direction.copy(),
                gripper_opening=self.plate_thickness + 0.2,  # Add small margin
            )
            grasps.append(grasp)
        
        return grasps
    
    def generate_all_grasps(
        self,
        num_grasps_per_edge: int = 5,
        include_corners: bool = True,
    ) -> List[GraspPose]:
        """
        Generate all possible grasp poses around the plate edges.
        
        All poses are in the plate frame (centroid at origin).
        
        Args:
            num_grasps_per_edge: Number of grasps to generate per edge
            include_corners: If True, add extra grasps at corners
        
        Returns:
            List of all GraspPose objects (all in plate frame)
        """
        all_grasps = []
        
        # Define the four edges of the plate
        # Frame is at center (0, 0, 0)
        # Plate extends from -half_width to +half_width in x
        # and from -half_length to +half_length in y
        # Thickness is in z direction
        
        # Edge 1: Left edge (x = -half_width, y varies, z = 0)
        # Approach from -x direction
        edge1_center = np.array([-self.half_width, 0.0, 0.0])
        edge1_direction = np.array([0.0, 1.0, 0.0])  # Along y-axis
        approach1 = np.array([-1.0, 0.0, 0.0])  # Approach from -x
        gripper_axis1 = np.array([0.0, 0.0, 1.0])  # Gripper opens in z direction
        grasps1 = self.generate_grasps_along_edge(
            edge1_center, edge1_direction, self.plate_length,
            approach1, num_grasps_per_edge, gripper_axis1
        )
        all_grasps.extend(grasps1)
        
        # Edge 2: Right edge (x = +half_width, y varies, z = 0)
        # Approach from +x direction
        edge2_center = np.array([self.half_width, 0.0, 0.0])
        edge2_direction = np.array([0.0, 1.0, 0.0])  # Along y-axis
        approach2 = np.array([1.0, 0.0, 0.0])  # Approach from +x
        gripper_axis2 = np.array([0.0, 0.0, 1.0])  # Gripper opens in z direction
        grasps2 = self.generate_grasps_along_edge(
            edge2_center, edge2_direction, self.plate_length,
            approach2, num_grasps_per_edge, gripper_axis2
        )
        all_grasps.extend(grasps2)
        
        # Edge 3: Front edge (y = -half_length, x varies, z = 0)
        # Approach from -y direction
        edge3_center = np.array([0.0, -self.half_length, 0.0])
        edge3_direction = np.array([1.0, 0.0, 0.0])  # Along x-axis
        approach3 = np.array([0.0, -1.0, 0.0])  # Approach from -y
        gripper_axis3 = np.array([0.0, 0.0, 1.0])  # Gripper opens in z direction
        grasps3 = self.generate_grasps_along_edge(
            edge3_center, edge3_direction, self.plate_width,
            approach3, num_grasps_per_edge, gripper_axis3
        )
        all_grasps.extend(grasps3)
        
        # Edge 4: Back edge (y = +half_length, x varies, z = 0)
        # Approach from +y direction
        edge4_center = np.array([0.0, self.half_length, 0.0])
        edge4_direction = np.array([1.0, 0.0, 0.0])  # Along x-axis
        approach4 = np.array([0.0, 1.0, 0.0])  # Approach from +y
        gripper_axis4 = np.array([0.0, 0.0, 1.0])  # Gripper opens in z direction
        grasps4 = self.generate_grasps_along_edge(
            edge4_center, edge4_direction, self.plate_width,
            approach4, num_grasps_per_edge, gripper_axis4
        )
        all_grasps.extend(grasps4)
        
        # Add corner grasps if requested
        if include_corners:
            corners = [
                np.array([-self.half_width, -self.half_length, 0.0]),  # Front-left
                np.array([self.half_width, -self.half_length, 0.0]),   # Front-right
                np.array([-self.half_width, self.half_length, 0.0]),   # Back-left
                np.array([self.half_width, self.half_length, 0.0]),    # Back-right
            ]
            
            for corner in corners:
                # Approach from diagonal direction (45 degrees)
                approach = corner / np.linalg.norm(corner)
                position = corner + self.grasp_offset * approach
                z_axis = -approach
                y_axis = np.array([0.0, 0.0, 1.0])  # Gripper opens in z
                y_axis = y_axis - np.dot(y_axis, z_axis) * z_axis
                y_axis = y_axis / np.linalg.norm(y_axis)
                x_axis = np.cross(y_axis, z_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)
                y_axis = np.cross(z_axis, x_axis)
                
                orientation = self._rotation_matrix_from_axes(x_axis, y_axis, z_axis)
                
                corner_grasp = GraspPose(
                    position=position,
                    orientation=orientation,
                    approach_direction=approach,
                    gripper_opening=self.plate_thickness + 0.2,
                )
                all_grasps.append(corner_grasp)
        
        return all_grasps
    
    def visualize_grasp_info(self, grasp: GraspPose) -> str:
        """Return a string representation of a grasp pose."""
        return (
            f"Position: [{grasp.position[0]:.2f}, {grasp.position[1]:.2f}, {grasp.position[2]:.2f}]\n"
            f"Approach: [{grasp.approach_direction[0]:.2f}, {grasp.approach_direction[1]:.2f}, {grasp.approach_direction[2]:.2f}]\n"
            f"Gripper opening: {grasp.gripper_opening:.2f} cm"
        )


def main():
    """Example usage of the grasp generator."""
    # Create generator
    generator = PlateGraspGenerator(
        plate_width=10.0,
        plate_length=20.0,
        plate_thickness=2.0,
        max_gripper_opening=5.0,
    )
    
    # Generate all grasps
    grasps = generator.generate_all_grasps(num_grasps_per_edge=5, include_corners=True)
    
    print(f"Generated {len(grasps)} grasp poses\n")
    
    # Print first few grasps as examples
    for i, grasp in enumerate(grasps[:5]):
        print(f"Grasp {i+1}:")
        print(generator.visualize_grasp_info(grasp))
        print()
    
    return grasps


if __name__ == "__main__":
    main()
