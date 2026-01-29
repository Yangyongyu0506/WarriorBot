from __future__ import annotations

"""Termination conditions for motion tracking and humanoid stability.

Defines fail conditions based on anchor position/orientation errors and body
deviations, supporting clean episode termination on unrecoverable states.
"""

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from booster_train.tasks.manager_based.beyond_mimic.mdp.commands import MotionCommand
from booster_train.tasks.manager_based.beyond_mimic.mdp.rewards import _get_body_indexes


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Terminate when global anchor position error exceeds `threshold`."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Terminate when vertical anchor position error exceeds `threshold`."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    """Terminate when anchor orientation deviates in gravity alignment by `threshold`."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Terminate when any tracked body position error exceeds `threshold`."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Terminate when any tracked body vertical position error exceeds `threshold`."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)

def is_fallen(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    foot_names: str | list[str] = ".*_foot_link",
    min_clearance: float = 0.18,
    tilt_threshold: float = 0.9,
    persist_steps: int = 3,
) -> torch.Tensor:
    """Terminate when the humanoid base drops too low or tilts too much.

    Parameters
    - asset_cfg: scene entity selecting the articulated robot and base body name (e.g., "Waist").
    - height_threshold: minimum acceptable base height (meters).
    - tilt_threshold: maximum acceptable tilt angle (radians). Approximate via projected gravity.

    Returns a boolean tensor of shape (num_envs,) indicating termination.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve body index for the specified base body (support regex names)
    if asset_cfg.body_ids == slice(None):
        ids, _ = asset.find_bodies(asset_cfg.body_names, preserve_order=True)
        base_idx = ids[0] if len(ids) > 0 else 0
    else:
        base_idx = asset_cfg.body_ids if isinstance(asset_cfg.body_ids, int) else int(asset_cfg.body_ids[0])

    # base height check relative to local foot height (robust on stairs)
    base_height = asset.data.body_pos_w[:, base_idx, 2]
    foot_ids, _ = asset.find_bodies(foot_names, preserve_order=True)
    if len(foot_ids) == 0:
        # fallback: treat no-foot match as not fallen
        relative_low = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        foot_z = asset.data.body_pos_w[:, torch.tensor(foot_ids, device=env.device), 2]
        min_foot_z = foot_z.min(dim=1).values
        relative_low = base_height < (min_foot_z + min_clearance)

    # tilt check via projected gravity onto base frame (z component close to 1.0 when upright)
    base_quat = asset.data.body_quat_w[:, base_idx]
    projected_gravity_b = math_utils.quat_apply_inverse(base_quat, asset.data.GRAVITY_VEC_W)
    # Acceptable tilt corresponds to z component >= cos(tilt_threshold)
    min_z = math.cos(tilt_threshold)
    too_tilted = projected_gravity_b[:, 2] < min_z

    # require both low clearance and significant tilt to avoid false positives
    fallen_now = torch.logical_and(relative_low, too_tilted)

    # persistence: only terminate if condition holds for `persist_steps` consecutive steps
    if not hasattr(env, "_fallen_counter"):
        env._fallen_counter = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
    env._fallen_counter = torch.where(fallen_now, env._fallen_counter + 1, torch.zeros_like(env._fallen_counter))
    return env._fallen_counter >= persist_steps