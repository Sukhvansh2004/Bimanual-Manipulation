#!/usr/bin/env python3
"""MeshCat TF visualizer (RViz-like).

Continuously publishes all known frames from TransformManager into MeshCat.

Prereqs:
- Start a meshcat server in another terminal: `meshcat-server`
  (default: websocket/ui on http://127.0.0.1:7000, ZMQ bridge on tcp://127.0.0.1:6000)

Usage:
  python scripts/transforms/transform_visualizer_meshcat.py

Stop:
  Ctrl+C
"""

import argparse
import time
import threading

import numpy as np
import meshcat.geometry as g

from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.transforms import TransformManager
from grasp_gen.utils.meshcat_utils import create_visualizer, make_frame

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize TransformManager frames in MeshCat")
    parser.add_argument("--fps", type=float, default=20.0, help="Update rate (Hz)")
    parser.add_argument(
        "--zmq-url",
        type=str,
        default="tcp://127.0.0.1:6000",
        help="MeshCat ZMQ URL (match meshcat-server)",
    )
    parser.add_argument("--axis-length", type=float, default=0.12, help="Axis triad length")
    parser.add_argument("--axis-radius", type=float, default=0.008, help="Axis triad radius")
    parser.add_argument("--labels", action="store_true", help="Render frame name labels in the 3D view")
    parser.add_argument("--label-offset", type=float, default=0.03, help="Z offset (meters) for label placement")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear visualizer on start")
    args = parser.parse_args()

    try:
        import meshcat
    except Exception as e:
        raise SystemExit(
            f"meshcat is not installed/available ({e}). Install it (e.g. `pip install meshcat`) and run `meshcat-server`."
        )

    # Prefer GraspGen helper (it matches the repo's meshcat-server setup), otherwise create directly.
    if create_visualizer is not None:
        vis = create_visualizer(clear=(not args.no_clear))
    else:
        vis = meshcat.Visualizer(zmq_url=args.zmq_url)
        if not args.no_clear:
            vis.delete()

        # Keep the UX minimal: tell user how to start / open meshcat.
        print("Waiting for meshcat server... start one with: meshcat-server")
        print("Then open: http://127.0.0.1:7000")

    # Create environment (headless; no MuJoCo window required)
    env_lock = threading.Lock()
    env = make_dual_arm_env()

    tfm = TransformManager(env=env, auto_discover=True, env_lock=env_lock)


    def _make_label_material(text: str):

        # Render text into an RGBA image
        pad = 6
        font = ImageFont.load_default()
        # Approximate size
        bbox = font.getbbox(text)
        text_w = max(1, bbox[2] - bbox[0])
        text_h = max(1, bbox[3] - bbox[1])
        img_w = text_w + 2 * pad
        img_h = text_h + 2 * pad

        img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, img_w - 1, img_h - 1], fill=(0, 0, 0, 140))
        draw.text((pad, pad), text, font=font, fill=(255, 255, 255, 255))

        texture = g.ImageTexture(image=img)
        material = g.MeshBasicMaterial(map=texture, transparent=True, opacity=1.0)
        return material, img_w, img_h

    def _ensure_frame_drawn(frame_id: str, drawn: set[str]) -> None:
        name = f"frames/{frame_id}"
        if name in drawn:
            return

        if make_frame is not None:
            make_frame(vis, name=name, h=float(args.axis_length), radius=float(args.axis_radius), o=0.9)
        else:
            vis[name].set_object(g.Axes(scale=float(args.axis_length)))

        drawn.add(name)

    def _ensure_label_drawn(frame_id: str, drawn_labels: set[str]) -> None:
        if not args.labels:
            return
        name = f"labels/{frame_id}"
        if name in drawn_labels:
            return
        if g is None:
            return

        mat_info = _make_label_material(frame_id)
        if mat_info is None:
            return
        material, img_w, img_h = mat_info

        # Map pixels to approximate metric size; keep small and readable.
        # Width scales with name length.
        w = 0.01 + 0.0025 * max(1, len(frame_id))
        h = 0.02
        d = 0.001

        vis[name].set_object(g.Box([w, h, d]), material)
        drawn_labels.add(name)

    drawn: set[str] = set()
    drawn_labels: set[str] = set()

    # Draw initial frames
    for frame_id in tfm.get_known_frames():
        _ensure_frame_drawn(frame_id, drawn)
        _ensure_label_drawn(frame_id, drawn_labels)

    dt = 1.0 / max(float(args.fps), 1e-6)
    print("Streaming TF frames to MeshCat. Ctrl+C to stop.")

    try:
        while True:
            # Refresh sim state (and guard MuJoCo access)
            with env_lock:
                try:
                    env.sim.forward()
                except Exception:
                    pass

            # Add any newly discovered frames (e.g., object_* frames)
            for frame_id in tfm.get_known_frames():
                _ensure_frame_drawn(frame_id, drawn)
                _ensure_label_drawn(frame_id, drawn_labels)

            # Publish transforms
            for frame_id in tfm.get_known_frames():
                try:
                    pose_w = tfm.get_frame_pose(frame_id, reference_frame="world")
                    T = pose_w.to_matrix().astype(float)
                    vis[f"frames/{frame_id}"].set_transform(T)

                    if args.labels and g is not None:
                        # Keep label orientation fixed in world for readability.
                        Tl = np.eye(4)
                        Tl[:3, 3] = np.asarray(pose_w.position, dtype=float) + np.array([0.0, 0.0, float(args.label_offset)])
                        vis[f"labels/{frame_id}"].set_transform(Tl)
                except Exception:
                    # Frame might be temporarily unavailable; skip
                    continue

            time.sleep(dt)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
