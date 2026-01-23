#!/usr/bin/env python3
"""MeshCat TF visualizer demo (RViz-like).

This is a thin wrapper around the reusable transforms module API.
It creates its own environment as a demo and streams that env's TF.

Prereqs:
- Start a meshcat server in another terminal: `meshcat-server`
  (default: websocket/ui on http://127.0.0.1:7000, ZMQ bridge on tcp://127.0.0.1:6000)

Usage:
  python scripts/transforms/transform_visualizer_meshcat.py

Stop:
  Ctrl+C
"""

import argparse
import threading

from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.transforms import MeshcatTFVisualizerConfig, stream_transforms_to_meshcat


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
    parser.add_argument(
        "--labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render frame name labels in the 3D view",
    )
    parser.add_argument("--label-offset", type=float, default=0.05, help="Z offset (meters) for label placement")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear visualizer on start")
    args = parser.parse_args()

    # Create environment
    env = make_dual_arm_env(robots=["Panda", "Panda"])
    
    # Create a reentrant lock to guard the environment
    # RLock is crucial because TransformManager might try to acquire it
    # while we already hold it in a loop.
    env_lock = threading.RLock()

    config = MeshcatTFVisualizerConfig(
        fps=float(args.fps),
        zmq_url=str(args.zmq_url),
        clear=not bool(args.no_clear),
        axis_length=float(args.axis_length),
        axis_radius=float(args.axis_radius),
        labels=bool(args.labels),
        label_offset=float(args.label_offset),
    )

    try:
        stream_transforms_to_meshcat(env=env, env_lock=env_lock, auto_discover=True, config=config)
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
