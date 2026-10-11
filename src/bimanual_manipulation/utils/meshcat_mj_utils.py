"""MuJoCo-to-MeshCat geometry conversion and scene publishing utilities."""

import numpy as np
import trimesh
import meshcat
import meshcat.geometry as g
import logging
from typing import Dict, Optional, Set

from robosuite.utils.binding_utils import MjSim, MjModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone geometry converter (shared by all MeshCat visualizer classes)
# ---------------------------------------------------------------------------

def mujoco_geom_to_meshcat(
    model,
    geom_id: int,
    mesh_cache: Dict[str, g.Geometry],
    mesh_sample_ratio: float = 1.0,
) -> Optional[g.Geometry]:
    """Convert a MuJoCo geom to a MeshCat geometry, with caching.

    Handles sphere, capsule (approximated as a cylinder without caps),
    cylinder, box, and mesh geom types. Cylinders/capsules are built along
    MeshCat's Y axis; callers must rotate them to MuJoCo's Z axis.
    Results are cached in *mesh_cache* (keyed by ``geom_<id>_type_<t>``;
    ``mesh_sample_ratio`` is not part of the key).

    Args:
        model: MuJoCo model (native or robosuite wrapper).
        geom_id: Index of the geom in the model.
        mesh_cache: Mutable dict used as a geometry cache.
        mesh_sample_ratio: Fraction of mesh faces to keep (0–1).
            Values < 1 subsample faces (evenly spaced, at least 4) for
            meshes with more than 10 faces to reduce MeshCat rendering load.

    Returns:
        A ``meshcat.geometry.Geometry`` instance, or *None* for unsupported types.
    """
    geom_type = model.geom_type[geom_id]
    cache_key = f"geom_{geom_id}_type_{geom_type}"

    if cache_key in mesh_cache:
        return mesh_cache[cache_key]

    geometry: Optional[g.Geometry] = None

    if geom_type == 7:  # Mesh
        mesh_id = model.geom_dataid[geom_id]
        vert_adr = model.mesh_vertadr[mesh_id]
        vert_num = model.mesh_vertnum[mesh_id]
        face_adr = model.mesh_faceadr[mesh_id]
        face_num = model.mesh_facenum[mesh_id]

        vertices = np.array(model.mesh_vert[vert_adr:vert_adr + vert_num])
        faces = np.array(model.mesh_face[face_adr:face_adr + face_num])

        if mesh_sample_ratio < 1.0 and len(faces) > 10:
            num_keep = max(4, int(len(faces) * mesh_sample_ratio))
            idx = np.linspace(0, len(faces) - 1, num_keep, dtype=int)
            faces = faces[idx]

        geometry = g.TriangularMeshGeometry(vertices, faces)

    elif geom_type == 2:  # Sphere
        radius = float(model.geom_size[geom_id][0])
        geometry = g.Sphere(radius)

    elif geom_type == 3:  # Capsule (approximated as cylinder)
        radius = float(model.geom_size[geom_id][0])
        height = float(model.geom_size[geom_id][1]) * 2
        geometry = g.Cylinder(height, radius)

    elif geom_type == 5:  # Cylinder
        radius = float(model.geom_size[geom_id][0])
        height = float(model.geom_size[geom_id][1]) * 2
        geometry = g.Cylinder(height, radius)

    elif geom_type == 6:  # Box
        size = np.array(model.geom_size[geom_id]) * 2
        geometry = g.Box(size.tolist())

    if geometry is not None:
        mesh_cache[cache_key] = geometry

    return geometry


class MeshcatMjVisualizer:
    """Utility to publish MuJoCo environment geometries to MeshCat.

    Attributes:
        vis: Target MeshCat visualizer.
    """
    def __init__(self, vis: meshcat.Visualizer):
        """Initialize the visualizer.

        Args:
            vis: MeshCat visualizer to publish into.
        """
        self.vis = vis
        # Cache for MeshCat geometries to avoid re-converting MuJoCo geoms
        self._mesh_cache: Dict[str, g.Geometry] = {}
        # Track published geoms to avoid redundant set_object calls
        self._published_geoms: Set[str] = set()

    def _get_meshcat_geometry(self, model: MjModel, geom_id: int) -> Optional[g.Geometry]:
        """Convert a MuJoCo geom to MeshCat geometry using the instance cache.

        Args:
            model: MuJoCo model.
            geom_id: Geom index.

        Returns:
            MeshCat geometry, or ``None`` for unsupported geom types.
        """
        return mujoco_geom_to_meshcat(model, geom_id, self._mesh_cache)

    def _get_group_for_geom(self, geom_name: str) -> str:
        """Determine the visualization group for a geometry based on its name prefix.

        Args:
            geom_name: Geom name.

        Returns:
            Text before the first underscore, or ``"other"`` if the name has
            no underscore.
        """
        # Generalize grouping: use the first part of the name (before the first underscore)
        # e.g., 'robot0_link1' -> 'robot0', 'wall_left' -> 'wall'
        if "_" in geom_name:
            return geom_name.split('_')[0]
        return "other"

    def publish_environment(self, sim: MjSim):
        """Publish geoms in group 0 or 1 to MeshCat and update their poses.

        Geoms whose name contains ``"floor"`` or ``"ground"`` are skipped.
        Objects are created once under ``scene/<group>/<geom_name>``; world
        transforms are set on every call. Per-geom errors are logged and
        skipped.

        Args:
            sim: Simulation whose current ``data`` geom poses are published.
        """
        model = sim.model
        data = sim.data
        
        # logger.debug(f"Publishing environment. Total geoms: {model.ngeom}")
        
        for geom_id in range(model.ngeom):
            try:
                # Only visualize geoms in group 0 (collision) or 1 (visual)
                # Typically group 1 is used for visual meshes in robosuite
                geom_group = model.geom_group[geom_id]
                geom_name = model.geom_id2name(geom_id)
                if not geom_name:
                    geom_name = f"geom_{geom_id}"

                # Log geom details for debugging
                # logger.debug(f"Geom {geom_id}: name='{geom_name}', group={geom_group}, type={model.geom_type[geom_id]}")

                if geom_group not in [0, 1]:
                    # logger.debug(f"Skipping geom {geom_name} (group {geom_group} not in [0, 1])")
                    continue
                    
                
                # Filter for robot/object geoms if needed, or just publish everything
                # For now, publish everything except the floor/arena if it's too big
                if "floor" in geom_name or "ground" in geom_name:
                    continue
                
                # Determine group based on name
                group = self._get_group_for_geom(geom_name)
                path = f"scene/{group}/{geom_name}"
                
                # Get geometry
                geometry = self._get_meshcat_geometry(model, geom_id)
                if geometry is None:
                    # logger.debug(f"Skipping geom {geom_name} (geometry creation failed)")
                    continue
                    
                # Publish object if not already done
                if path not in self._published_geoms:
                    # logger.debug(f"Publishing new geom: {path}")
                    rgba = model.geom_rgba[geom_id]
                    material = g.MeshPhongMaterial(
                        color=self._rgb2hex(rgba[:3]),
                        opacity=float(rgba[3]),
                        transparent=bool(rgba[3] < 1.0)
                    )
                    self.vis[path].set_object(geometry, material)
                    self._published_geoms.add(path)
                
                # Update transform
                pos = data.geom_xpos[geom_id]
                rot_mat = data.geom_xmat[geom_id].reshape(3, 3)
                T = np.eye(4)
                T[:3, :3] = rot_mat
                T[:3, 3] = pos
                
                # Adjust for cylinder/capsule orientation difference between MuJoCo and MeshCat
                geom_type = model.geom_type[geom_id]
                if geom_type in [3, 5]: # Capsule (3) or Cylinder (5)
                    # Rotate 90 deg around X to align Y-axis with Z-axis
                    R_align = np.array([
                        [1, 0, 0, 0],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]
                    ])
                    T = T @ R_align

                self.vis[path].set_transform(T)
            
            except Exception as e:
                logger.error(f"Error publishing geom {geom_id}: {e}")
                continue

    def _rgb2hex(self, rgb):
        """Convert RGB in ``[0, 1]`` to a hex string.

        Args:
            rgb: Array-like of 3 floats in ``[0, 1]``.

        Returns:
            String of the form ``"0xRRGGBB"``.
        """
        # Convert to python int for serialization safety
        vals = (np.array(rgb) * 255).astype(int).tolist()
        return "0x%02x%02x%02x" % tuple(vals)