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

from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.transforms import TransformManager


def _try_import_meshcat_utils():
    """Use GraspGen's meshcat helpers if available (already a dependency here)."""
    try:
        from grasp_gen.utils.meshcat_utils import create_visualizer, make_frame  # type: ignore

        return create_visualizer, make_frame
    except Exception:
        return None, None


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
    parser.add_argument("--no-clear", action="store_true", help="Do not clear visualizer on start")
    args = parser.parse_args()

    create_visualizer, make_frame = _try_import_meshcat_utils()

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

    def _ensure_frame_drawn(frame_id: str, drawn: set[str]) -> None:
        name = f"frames/{frame_id}"
        if name in drawn:
            return

        if make_frame is not None:
            make_frame(vis, name=name, h=float(args.axis_length), radius=float(args.axis_radius), o=0.9)
        else:
            # Minimal fallback: use MeshCat's built-in axes object
            import meshcat.geometry as g

            vis[name].set_object(g.Axes(scale=float(args.axis_length)))

        drawn.add(name)

    drawn: set[str] = set()

    # Draw initial frames
    for frame_id in tfm.get_known_frames():
        _ensure_frame_drawn(frame_id, drawn)

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

            # Publish transforms
            for frame_id in tfm.get_known_frames():
                try:
                    pose_w = tfm.get_frame_pose(frame_id, reference_frame="world")
                    T = pose_w.to_matrix().astype(float)
                    vis[f"frames/{frame_id}"].set_transform(T)
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
