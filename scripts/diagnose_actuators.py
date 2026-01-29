#!/usr/bin/env python3

"""
Actuator effectiveness diagnostic for Booster T1.

This script launches the manager-based env, applies non-zero actions to selected
joints, and reports changes in joint positions and torques over several steps.
Use it to verify that the action space drives the robot motion.
"""

import argparse
import time
import numpy as np
import torch

import gymnasium as gym

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose actuator effectiveness")
    parser.add_argument("--task", type=str, default="Template-Booster-T1-Rough-v0", help="Registered env id")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps to run")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel envs")
    parser.add_argument(
        "--joint_regex",
        type=str,
        default=(
            "Left_Hip_Pitch|Left_Knee_Pitch|Left_Ankle_Roll|"
            "Right_Hip_Pitch|Right_Knee_Pitch|Right_Ankle_Roll"
        ),
        help="Regex for joints to stimulate",
    )
    parser.add_argument("--action_value", type=float, default=0.25, help="Action amplitude to apply")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g., cuda)")
    return parser.parse_args()


def main():
    args = parse_args()

    # launch omniverse app (headless by default)
    app_launcher = AppLauncher()
    simulation_app = app_launcher.app

    # import env cfg after app launch
    from booster_t1_rough.tasks.manager_based.booster_t1_rough.booster_t1_rough_env_cfg import (
        BoosterT1RoughEnvCfg_PLAY,
    )

    # construct env cfg
    env_cfg = BoosterT1RoughEnvCfg_PLAY()
    env_cfg.scene.num_envs = args.num_envs
    if args.device is not None:
        env_cfg.sim.device = args.device

    # create env
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    robot = env.unwrapped.scene["robot"]

    # resolve joint ids by regex
    try:
        joint_ids, joint_names = robot.find_joints(args.joint_regex, preserve_order=True)
    except Exception as e:
        print("Failed to resolve joint regex.")
        print(f"Regex: {args.joint_regex}")
        print(f"Available joints: {robot.joint_names}")
        raise
    print(f"Selected joints ({len(joint_ids)}): {joint_names}")

    # action space
    # total action dimension across all terms
    total_action_dim = getattr(env.unwrapped.action_manager, "total_action_dim", None)
    if total_action_dim is None:
        # fallback to gym space
        shp = getattr(env.action_space, "shape", None)
        total_action_dim = int(np.prod(shp)) if shp is not None else None
    if total_action_dim is None:
        raise RuntimeError("Could not determine total action dimension.")
    print(f"Action space size: {total_action_dim}")

    # map robot joint names to action indices for the joint position term
    action_term = getattr(env.unwrapped.action_manager, "get_term", None)
    joint_pos_term = None
    controlled_joint_ids = None
    controlled_joint_names = None
    if action_term is not None:
        joint_pos_term = env.unwrapped.action_manager.get_term("joint_pos")
        controlled_joint_ids = getattr(joint_pos_term, "joint_ids", None)
        if controlled_joint_ids is None:
            controlled_joint_ids = getattr(joint_pos_term, "_joint_ids", None)
        if controlled_joint_ids is not None:
            # handle slice(None) meaning 'all joints'
            if isinstance(controlled_joint_ids, slice):
                controlled_joint_ids = list(range(len(robot.joint_names)))
                controlled_joint_names = robot.joint_names
            else:
            # convert to Python list
                try:
                    controlled_joint_ids = list(controlled_joint_ids)
                except Exception:
                    controlled_joint_ids = [int(x) for x in controlled_joint_ids]
                # derive names via robot articulation
                try:
                    controlled_joint_names = [robot.joint_names[i] for i in controlled_joint_ids]
                except Exception:
                    controlled_joint_names = None
        else:
            print("Warning: could not fetch controlled joint ids; will attempt direct index use.")

    # baseline step (zeros) to gather initial state
    obs, info = env.reset()
    # actions must be a torch.Tensor on the env device
    actions = torch.zeros((env.unwrapped.num_envs, total_action_dim), dtype=torch.float32, device=env.unwrapped.device)

    def stats(tag: str):
        jp = robot.data.joint_pos.clone().cpu().numpy()
        jv = robot.data.joint_vel.clone().cpu().numpy()
        # joint efforts may be unavailable depending on models; guard
        je = getattr(robot.data, "joint_effort", None)
        je_np = je.clone().cpu().numpy() if je is not None else None
        print(
            f"[{tag}] pos mean {jp.mean():.4f}, vel mean {jv.mean():.4f},"
            + (f" effort mean {je_np.mean():.4f}" if je_np is not None else " effort n/a")
        )

    stats("reset")

    # step a few times with zeros
    for _ in range(10):
        obs, rew, term, trunc, info = env.step(actions)
    stats("zero-actions-10-steps")

    # apply non-zero actions on selected joint indices
    stimulated = []
    # build name->index mapping if available
    name_to_index = {}
    if controlled_joint_names is not None:
        for idx, name in enumerate(controlled_joint_names):
            name_to_index[name] = idx
        print(f"Controlled joints in action term: {len(controlled_joint_names)}")
        print(f"First 8 controlled joints: {controlled_joint_names[:8]}")
    for env_idx in range(env.unwrapped.num_envs):
        for j_id, j_name in zip(joint_ids, joint_names):
            a_idx = None
            if name_to_index:
                a_idx = name_to_index.get(j_name, None)
            elif controlled_joint_ids is not None:
                # fallback by id match
                try:
                    # use list index to avoid numpy 0d warnings
                    a_idx = controlled_joint_ids.index(int(j_id))
                except Exception:
                    a_idx = None
            else:
                # naive fallback: assume direct id-index mapping
                a_idx = int(j_id) if int(j_id) < total_action_dim else None
            if a_idx is not None and 0 <= a_idx < total_action_dim:
                actions[env_idx, a_idx] = torch.tensor(args.action_value, dtype=torch.float32, device=actions.device)
                stimulated.append((a_idx, j_name))
    if stimulated:
        print("Stimulated action indices and joints:")
        for a_idx, j_name in stimulated[:10]:
            print(f"  action[{a_idx}] -> {j_name}")
    else:
        print("No joints stimulated via mapping; applying fallback on first N indices.")
        n = min(len(joint_names), total_action_dim)
        for env_idx in range(env.unwrapped.num_envs):
            for a_idx in range(n):
                actions[env_idx, a_idx] = torch.tensor(args.action_value, dtype=torch.float32, device=actions.device)
                stimulated.append((a_idx, f"idx_{a_idx}"))
        print("Stimulated fallback indices:")
        print([idx for idx, _ in stimulated[:10]])

    # capture pre and post states for delta
    jp_before = robot.data.joint_pos.clone().cpu().numpy()
    # step with non-zero actions
    for _ in range(args.steps):
        obs, rew, term, trunc, info = env.step(actions)
    jp_after = robot.data.joint_pos.clone().cpu().numpy()

    delta = jp_after - jp_before
    delta_sel = delta[:, joint_ids]
    print(f"Delta joint pos (selected) mean {delta_sel.mean():.4f}, max {delta_sel.max():.4f}")
    stats("post-actions")

    # basic assertion: movement observed on targeted joints
    moved = np.abs(delta_sel).mean() > 1e-3
    if moved:
        print("Actuator response: OK (targeted joints moved under non-zero actions)")
    else:
        print("Actuator response: FAILED (no observable movement). Check action mapping/scale/actuator config.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
