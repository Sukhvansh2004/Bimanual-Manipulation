#!/usr/bin/env python3
"""Quick test of experiment config loading (including new sections)."""
from robosuite_franka_manipulation.experiment_config import load_experiment, list_experiments

exps = list_experiments()
print(f"Found {len(exps)} experiments:")
for e in exps:
    print(f"  {e}")
print()

for e in exps:
    cfg = load_experiment(e)
    print(f"{cfg.name}: {cfg.description}")
    print(f"  YCB: {len(cfg.ycb_objects)}, Custom: {len(cfg.custom_objects)}")
    print(f"  Object: {cfg.task.manipulated_object}")
    print(f"  Target: {cfg.task.target_pose.position.tolist()}")
    print(f"  Has start: {cfg.task.start_pose is not None}")
    print(f"  Steps: {cfg.planning.num_steps}")
    print(f"  Filter: {cfg.collision_filter.exclude_patterns}")
    # New sections
    print(f"  Viz mesh: {cfg.visualization.visualize_mesh}, zmq: {cfg.visualization.zmq_url}")
    print(f"  Output save: {cfg.output.save_trajectories}, dir: {cfg.output.output_dir}")
    print(f"  Exec vis: {cfg.execution.visualize}, smoothen: {cfg.execution.smoothen}")
    print(f"  Runtime debug: {cfg.runtime.debug}, workers: {cfg.runtime.num_workers}")
    print()

print("ALL OK")
