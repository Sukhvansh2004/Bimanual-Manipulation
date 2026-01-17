import numpy as np
import trimesh
import meshcat
import meshcat.geometry as g
import logging
from typing import Dict, Optional, Set

from robosuite.utils.binding_utils import MjSim

logger = logging.getLogger(__name__)

class MeshcatMjVisualizer:
    """
    Utility to publish MuJoCo environment geometries to MeshCat.
    """
    def __init__(self, vis: meshcat.Visualizer):
        self.vis = vis
        # Cache for trimesh objects to avoid re-loading/re-processing
        self._mesh_cache: Dict[str, g.Geometry] = {}
        # Track published geoms to avoid redundant set_object calls
        self._published_geoms: Set[str] = set()

    def _get_meshcat_geometry(self, model, geom_id: int) -> Optional[g.Geometry]:
        """Convert MuJoCo geom to MeshCat geometry."""
        geom_type = model.geom_type[geom_id]
        
        # Unique key for caching
        cache_key = f"type_{geom_type}_size_{model.geom_size[geom_id]}"
        
        if geom_type == 7:  # Mesh
            mesh_id = model.geom_dataid[geom_id]
            cache_key = f"mesh_{mesh_id}"
            if cache_key in self._mesh_cache:
                return self._mesh_cache[cache_key]
            
            # Extract vertices and faces from MuJoCo model
            vert_adr = model.mesh_vertadr[mesh_id]
            vert_num = model.mesh_vertnum[mesh_id]
            face_adr = model.mesh_faceadr[mesh_id]
            face_num = model.mesh_facenum[mesh_id]
            
            vertices = model.mesh_vert[vert_adr : vert_adr + vert_num]
            faces = model.mesh_face[face_adr : face_adr + face_num]
            
            meshcat_geom = g.TriangularMeshGeometry(vertices, faces)
            self._mesh_cache[cache_key] = meshcat_geom
            return meshcat_geom
        
        elif geom_type == 2:  # Sphere
            radius = model.geom_size[geom_id][0]
            if cache_key in self._mesh_cache:
                return self._mesh_cache[cache_key]
            meshcat_geom = g.Sphere(radius)
            self._mesh_cache[cache_key] = meshcat_geom
            return meshcat_geom
            
        elif geom_type == 3:  # Capsule
            radius = model.geom_size[geom_id][0]
            height = model.geom_size[geom_id][1] * 2
            if cache_key in self._mesh_cache:
                return self._mesh_cache[cache_key]
            # MeshCat doesn't have a native capsule, use a cylinder for now
            meshcat_geom = g.Cylinder(height, radius)
            self._mesh_cache[cache_key] = meshcat_geom
            return meshcat_geom
            
        elif geom_type == 5:  # Cylinder (MuJoCo type 5)
            radius = model.geom_size[geom_id][0]
            height = model.geom_size[geom_id][1] * 2
            if cache_key in self._mesh_cache:
                return self._mesh_cache[cache_key]
            meshcat_geom = g.Cylinder(height, radius)
            self._mesh_cache[cache_key] = meshcat_geom
            return meshcat_geom
            
        elif geom_type == 6:  # Box
            size = model.geom_size[geom_id] * 2
            if cache_key in self._mesh_cache:
                return self._mesh_cache[cache_key]
            meshcat_geom = g.Box(size)
            self._mesh_cache[cache_key] = meshcat_geom
            return meshcat_geom
            
        return None

    def _get_group_for_geom(self, geom_name: str) -> str:
        """Determine the visualization group for a geometry based on its name prefix."""
        # Generalize grouping: use the first part of the name (before the first underscore)
        # e.g., 'robot0_link1' -> 'robot0', 'wall_left' -> 'wall'
        if "_" in geom_name:
            return geom_name.split('_')[0]
        return "other"

    def publish_environment(self, sim: MjSim):
        """Publish all geoms in the simulation to MeshCat."""
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
        """Convert RGB [0, 1] to hex string."""
        # Convert to python int for serialization safety
        vals = (np.array(rgb) * 255).astype(int).tolist()
        return "0x%02x%02x%02x" % tuple(vals)