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
import meshcat
import meshcat.geometry as g
from grasp_gen.utils.meshcat_utils import create_visualizer, make_frame
from PIL import Image, ImageDraw, ImageFont

@dataclass(frozen=True)
class MeshcatTFVisualizerConfig:
    fps: float = 20.0
    zmq_url: str = "tcp://127.0.0.1:6000"
    clear: bool = True

    axis_length: float = 0.12
    axis_radius: float = 0.008

    labels: bool = True
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

    if create_visualizer is not None:
        vis = create_visualizer(clear=bool(config.clear))
    else:
        vis = meshcat.Visualizer(zmq_url=str(config.zmq_url))
        if config.clear:
            vis.delete()

        print("Waiting for meshcat server... start one with: meshcat-server")
        print("Then open: http://127.0.0.1:7000")

    tfm = tf_manager or TransformManager(env=env, auto_discover=auto_discover, env_lock=env_lock)

    def _make_label_material(text: str):
        if not config.labels:
            return None

        pad = 6
        font = ImageFont.load_default()
        bbox = font.getbbox(text)
        text_w = max(1, bbox[2] - bbox[0])
        text_h = max(1, bbox[3] - bbox[1])
        img_w = text_w + 2 * pad
        img_h = text_h + 2 * pad

        img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, img_w - 1, img_h - 1], fill=(0, 0, 0, 140))
        draw.text((pad, pad), text, font=font, fill=(255, 255, 255, 255))

        # meshcat expects a meshcat.geometry.Image subclass (e.g., PngImage), not a PIL Image.
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = g.PngImage(buf.getvalue())
        texture = g.ImageTexture(image=png)
        material = g.MeshBasicMaterial(color=0xFFFFFF, map=texture, transparent=True, opacity=1.0)
        return material, img_w, img_h

    def _ensure_frame_drawn(frame_id: str, drawn: set[str]) -> None:
        name = f"frames/{frame_id}"
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

        name = f"labels/{frame_id}"
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
    print("Streaming TF frames to MeshCat. Ctrl+C to stop.")

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
                    vis[f"frames/{frame_id}"].set_transform(T)

                    if config.labels:
                        Tl = np.eye(4)
                        Tl[:3, 3] = np.asarray(pose_w.position, dtype=float) + np.array(
                            [0.0, 0.0, float(config.label_offset)],
                            dtype=float,
                        )
                        vis[f"labels/{frame_id}"].set_transform(Tl)
                except Exception:
                    continue

            time.sleep(dt)

    except KeyboardInterrupt:
        pass
