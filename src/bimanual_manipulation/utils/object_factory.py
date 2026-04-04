"""Object creation, mesh extraction, and YCB dataset utilities.

Provides helpers for creating object specifications (YCB and primitive),
extracting meshes from the MuJoCo simulation, and resolving mesh file paths.
"""

import atexit
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import trimesh

from bimanual_manipulation import constants
from bimanual_manipulation.environments.dual_arm import DualArmEnvironment

logger = logging.getLogger(__name__)


# =============================================================================
# Global Mesh Extraction Registry
# =============================================================================


class MeshExtractionManager:
    """Manages lazy extraction of meshes from environment and automatic cleanup."""

    def __init__(self):
        self.env = None
        self.temp_dir = None

    def register_env(self, env: DualArmEnvironment):
        self.env = env
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="meshes_")
            atexit.register(self.cleanup)
            logger.debug(f"Registered mesh extraction temp directory: {self.temp_dir}")

    def cleanup(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.debug(f"Cleaned up mesh extraction temp directory: {self.temp_dir}")
            self.temp_dir = None

    def get_mesh_path(self, object_name: str) -> Optional[str]:
        if self.temp_dir is None:
            return None

        mesh_path = os.path.join(self.temp_dir, f"{object_name}.obj")
        if os.path.exists(mesh_path):
            return mesh_path

        if self.env is not None:
            visual_geom = f"{object_name}_visual"
            success = extract_mesh_from_geom(self.env, visual_geom, mesh_path)
            if not success:
                success = extract_mesh_from_geom(self.env, object_name, mesh_path)

            if success:
                return mesh_path

        return None


_mesh_manager = MeshExtractionManager()


def register_mesh_extraction_env(env: DualArmEnvironment):
    """Register an environment for lazy mesh extraction."""
    _mesh_manager.register_env(env)


# =============================================================================
# MuJoCo Mesh Extraction Utilities
# =============================================================================


def extract_mesh_from_geom(env: DualArmEnvironment, geom_name: str, output_path: str) -> bool:
    """Extract mesh data from a MuJoCo geom and save it to a file.

    Args:
        env: Environment instance.
        geom_name: Name of the geom to extract.
        output_path: Path to save the mesh file.

    Returns:
        True if successful, False otherwise.
    """
    try:
        model = env.sim.model
        try:
            geom_id = model.geom_name2id(geom_name)
        except Exception:
            found = False
            for i in range(model.ngeom):
                name = model.geom_id2name(i)
                if name and geom_name in name:
                    geom_id = i
                    found = True
                    break
            if not found:
                return False

        geom_type = model.geom_type[geom_id]

        mesh = None
        if geom_type == 7:  # Mesh
            mesh_id = model.geom_dataid[geom_id]
            vert_adr = model.mesh_vertadr[mesh_id]
            vert_num = model.mesh_vertnum[mesh_id]
            face_adr = model.mesh_faceadr[mesh_id]
            face_num = model.mesh_facenum[mesh_id]

            vertices = model.mesh_vert[vert_adr : vert_adr + vert_num]
            faces = model.mesh_face[face_adr : face_adr + face_num]
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        elif geom_type == 6:  # Box
            size = model.geom_size[geom_id]
            mesh = trimesh.creation.box(extents=size * 2)

        elif geom_type == 5:  # Cylinder
            radius, half_height = model.geom_size[geom_id][:2]
            mesh = trimesh.creation.cylinder(radius=radius, height=half_height * 2)

        elif geom_type == 2:  # Sphere
            radius = model.geom_size[geom_id][0]
            mesh = trimesh.creation.icosphere(radius=radius)

        if mesh is not None:
            mesh.export(output_path)
            logger.info(f"Extracted mesh for geom '{geom_name}' to {output_path}")
            return True

        return False

    except Exception as e:
        logger.error(f"Failed to extract mesh for geom '{geom_name}': {e}")
        return False


# =============================================================================
# YCB Dataset Utilities
# =============================================================================


def get_available_ycb_objects() -> List[str]:
    """Get a sorted list of all available YCB objects."""
    if not constants.YCB_DATASET_PATH.exists():
        logger.warning(f"YCB dataset not found at {constants.YCB_DATASET_PATH}")
        return []

    objects = []
    for xml_file in constants.YCB_DATASET_PATH.glob("*.xml"):
        objects.append(xml_file.stem)
    return sorted(objects)


def list_ycb_objects() -> None:
    """Print all available YCB objects to console."""
    objects = get_available_ycb_objects()
    print(f"\nAvailable YCB Objects ({len(objects)} total):")
    print("-" * 40)
    for i, obj in enumerate(objects):
        print(f"  {i+1:3d}. {obj}")
    print("-" * 40)


def resolve_mesh_path(object_name: str) -> Optional[str]:
    """Resolve the mesh path for a given object name.

    Args:
        object_name: Name of the object (e.g., ``'apple'``, ``'apple_0'``).

    Returns:
        Path to the mesh file if found, ``None`` otherwise.
    """
    candidates = [object_name]
    if "_" in object_name:
        candidates.append(object_name.rsplit("_", 1)[0])

    for name in candidates:
        ycb_obj_dir = constants.YCB_DATASET_PATH / name
        if ycb_obj_dir.exists():
            mesh_path = ycb_obj_dir / "textured.obj"
            if mesh_path.exists():
                return str(mesh_path)

    return _mesh_manager.get_mesh_path(object_name)


# =============================================================================
# Object Specification Helpers
# =============================================================================


def create_ycb_object(
    ycb_name: str,
    instance_name: Optional[str] = None,
    position: Optional[List[float]] = None,
    quaternion: Optional[List[float]] = None,
    is_static: bool = False,
) -> Dict[str, Any]:
    """Create a YCB object specification for the environment.

    Args:
        ycb_name: Name of the YCB object (e.g., ``'apple'``, ``'mug'``).
        instance_name: Optional unique instance name (defaults to ``ycb_name``).
        position: Optional ``[x, y, z]`` position.
        quaternion: Optional ``[x, y, z, w]`` quaternion.
        is_static: Whether the object is static.

    Returns:
        Dictionary with object specification.
    """
    if instance_name is None:
        instance_name = f"{ycb_name}_0"

    return {
        "type": "ycb",
        "ycb_name": ycb_name,
        "instance_name": instance_name,
        "name": instance_name,
        "pos": position if position else constants.DEFAULT_OBJECT_POSITION.copy(),
        "quat": quaternion if quaternion else constants.DEFAULT_OBJECT_QUATERNION.copy(),
        "is_static": is_static,
    }


def create_primitive_object(
    obj_type: str,
    name: str,
    size: List[float],
    position: Optional[List[float]] = None,
    quat: Optional[List[float]] = None,
    rgba: Optional[List[float]] = None,
    density: float = 500.0,
    friction: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Create a primitive object specification for the environment.

    Args:
        obj_type: Type of primitive (``'box'``, ``'cylinder'``, ``'ball'``, ``'capsule'``).
        name: Unique name for the object.
        size: Size parameters (varies by type).
        position: Optional ``[x, y, z]`` position.
        quat: Optional ``[x, y, z, w]`` quaternion.
        rgba: Optional ``[r, g, b, a]`` color values.
        density: Object density in kg/m³.
        friction: Optional ``[slide, spin, roll]`` MuJoCo friction coefficients.

    Returns:
        Dictionary with object specification.
    """
    return {
        "type": obj_type,
        "name": name,
        "size": size,
        "pos": position if position else constants.DEFAULT_OBJECT_POSITION.copy(),
        "quat": quat if quat else [0.0, 0.0, 0.0, 1.0],
        "rgba": rgba if rgba else [0.5, 0.5, 0.5, 1.0],
        "friction": friction,
        "density": density,
    }
