"""MeshCat-based TF visualizer.

This module provides a reusable visualizer that publishes frames from a
:class:`~robosuite_franka_manipulation.transforms.transform_manager.TransformManager`
into MeshCat.

Design goal: let callers pass their *own* `env` + `env_lock` so the visualizer
reflects the same simulation instance they are working with.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional

import numpy as np
import io

from robosuite_franka_manipulation.transforms import TransformManager
from robosuite_franka_manipulation import constants
import meshcat
import meshcat.geometry as g
from grasp_gen.utils.meshcat_utils import make_frame
from PIL import Image, ImageDraw, ImageFont

import logging
from logging import getLogger

logging.basicConfig(level=constants.LOG_LEVEL)
logger = getLogger(__name__)


@dataclass(frozen=True)
class MeshcatTFVisualizerConfig:
    fps: float = constants.MESHCAT_TF_VISUALIZER_FPS
    zmq_url: str = constants.MESHCAT_ZMQ_URL
    clear: bool = True

    axis_length: float = 0.12
    axis_radius: float = 0.008

    labels: bool = True
    # World Z offset (meters) for label placement *below* the frame origin.
    label_offset: float = 0.05


def stream_transforms_to_meshcat(
    *,
    env,
    env_lock,
    tf_manager: Optional[TransformManager] = None,
    auto_discover: bool = True,
    config: MeshcatTFVisualizerConfig = MeshcatTFVisualizerConfig(),
    stop_event=None,
) -> None:
    """Continuously publish TF frames to MeshCat.

    Args:
        env: Robosuite environment instance.
        env_lock: A lock guarding access to MuJoCo state (e.g., `env.sim`).
        tf_manager: Optionally pass an existing TransformManager.
        auto_discover: If creating a TransformManager internally, enable auto-discovery.
        config: Visualizer configuration.
        stop_event: Optional threading.Event-like object; if provided, loop exits when set.
    """

    # Always honor config.zmq_url (and therefore constants.MESHCAT_ZMQ_URL).
    # The helper in grasp_gen hardcodes its own zmq url, so we avoid it here.
    vis = meshcat.Visualizer(zmq_url=str(config.zmq_url))
    if config.clear:
        try:
            vis.delete()
        except Exception:
            pass

    logger.info("Waiting for meshcat server... start one with: meshcat-server")
    logger.info("Then open: http://127.0.0.1:7000")

    tfm = tf_manager or TransformManager(env=env, auto_discover=auto_discover, env_lock=env_lock)

    def _group_for_frame(frame_id: str) -> str:
        if frame_id == "world":
            return "world"
            
        # Check for cameras first (including eye-in-hand which starts with robot prefix)
        if frame_id.endswith("_mj"):
            base = frame_id[:-3]
            if base.endswith("view") or "eye_in_hand" in base:
                return "cameras/raw"
        if frame_id.endswith("_optical"):
            base = frame_id[: -len("_optical")]
            if base.endswith("view") or "eye_in_hand" in base:
                return "cameras"
        if frame_id.endswith("view") or "eye_in_hand" in frame_id:
            return "cameras"
            
        # Then check for robots
        if frame_id.startswith("robot0_"):
            return "robots/robot0"
        if frame_id.startswith("robot1_"):
            return "robots/robot1"
        if frame_id.startswith("robot") and "_" in frame_id:
            # Any additional robots
            return "robots/other"
        if frame_id.startswith("object"):
            return "objects"
        if frame_id.startswith("grasp"):
            return "grasps"
            
        return "other"

    def _frame_path(frame_id: str) -> str:
        # Frame root node. Children under this path will toggle together.
        return f"frames/{_group_for_frame(frame_id)}/{frame_id}"

    def _axes_path(frame_id: str) -> str:
        return f"{_frame_path(frame_id)}/axes"

    def _label_path(frame_id: str) -> str:
        return f"{_frame_path(frame_id)}/label"

    def _make_label_material(text: str):
        if not config.labels:
            return None

        pad = 2
        font = ImageFont.load_default()
        bbox = font.getbbox(text)
        text_w = max(1, bbox[2] - bbox[0])
        text_h = max(1, bbox[3] - bbox[1])
        img_w = text_w + 2 * pad
        img_h = text_h + 2 * pad

        img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((pad, pad), text, font=font, fill=(255, 255, 255, 255))

        # meshcat expects a meshcat.geometry.Image subclass (e.g., PngImage), not a PIL Image.
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = g.PngImage(buf.getvalue())
        texture = g.ImageTexture(image=png)
        # Render only the front faces (avoid seeing the texture from both sides,
        # which can look like doubled / overlaid text).
        # three.js constants: FrontSide=0, BackSide=1, DoubleSide=2.
        material = g.MeshBasicMaterial(color=0xFFFFFF, map=texture, side=0, transparent=True, opacity=1.0)
        return material, img_w, img_h

    def _ensure_frame_drawn(frame_id: str, drawn: set[str]) -> None:
        name = _axes_path(frame_id)
        if name in drawn:
            return

        if make_frame is not None:
            make_frame(
                vis,
                name=name,
                h=float(config.axis_length),
                radius=float(config.axis_radius),
                o=0.9,
            )
        else:
            vis[name].set_object(g.Axes(scale=float(config.axis_length)))

        drawn.add(name)

    def _ensure_label_drawn(frame_id: str, drawn_labels: set[str]) -> None:
        if not config.labels:
            return

        name = _label_path(frame_id)
        if name in drawn_labels:
            return

        mat_info = _make_label_material(frame_id)
        if mat_info is None:
            return

        material, img_w, img_h = mat_info

        aspect = float(img_w) / max(float(img_h), 1.0)
        h = 0.03
        w = float(np.clip(h * aspect, 0.04, 0.18))
        d = 0.002

        vis[name].set_object(g.Box([w, h, d]), material)
        drawn_labels.add(name)

    drawn: set[str] = set()
    drawn_labels: set[str] = set()

    for frame_id in tfm.get_known_frames():
        _ensure_frame_drawn(frame_id, drawn)
        _ensure_label_drawn(frame_id, drawn_labels)

    dt = 1.0 / max(float(config.fps), 1e-6)
    logger.info("Streaming TF frames to MeshCat. Ctrl+C to stop.")

    # Keep labels upright (vertical) for readability.
    # Rotate +90deg about X so the label plane is not flat on XY.
    R_label = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    try:
        while True:
            if stop_event is not None and bool(getattr(stop_event, "is_set")()):
                break

            with env_lock:
                try:
                    env.sim.forward()
                except Exception:
                    pass

            for frame_id in tfm.get_known_frames():
                _ensure_frame_drawn(frame_id, drawn)
                _ensure_label_drawn(frame_id, drawn_labels)

            for frame_id in tfm.get_known_frames():
                try:
                    pose_w = tfm.get_frame_pose(frame_id, reference_frame="world")
                    T = pose_w.to_matrix().astype(float)
                    vis[_frame_path(frame_id)].set_transform(T)

                    if config.labels:
                        # Desired label pose in world.
                        Tl_world = np.eye(4)
                        Tl_world[:3, :3] = R_label
                        # Place label below the frame along world -Z.
                        Tl_world[:3, 3] = np.asarray(pose_w.position, dtype=float) - np.array(
                            [0.0, 0.0, float(config.label_offset)],
                            dtype=float,
                        )

                        # Label is nested under the frame node, so publish a frame-local transform.
                        Tl_local = np.linalg.inv(T) @ Tl_world
                        vis[_label_path(frame_id)].set_transform(Tl_local)
                except Exception:
                    continue

            time.sleep(dt)

    except KeyboardInterrupt:
        pass
