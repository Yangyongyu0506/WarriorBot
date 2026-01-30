from __future__ import annotations

"""Reward terms for humanoid motion tracking and gait quality.

This module defines exponential error rewards for tracking motion anchors,
body poses and velocities, as well as gait-specific terms such as feet
stance time. Rewards optionally use adaptive sigmas based on running error
statistics, stabilizing training across varying terrain difficulty.
"""

import torch
from typing import TYPE_CHECKING, Union

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv

from booster_train.tasks.manager_based.beyond_mimic.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    """Resolve indices of bodies to include according to `body_names` filter."""
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _get_adaptive_sigma(env, key: str | float, error: Union[float, torch.Tensor]):
    """Return scalar sigma from fixed value or an EMA of observed errors.

    If `key` is a string, maintains per-key error EMAs in the `env` to adapt
    the effective standard deviation used in exponential rewards.
    """
    if isinstance(key, float):
        return key
    sigma_update_rate = 0.9
    if not hasattr(env, 'reward_sigmas_ema'):
        env.reward_sigmas_ema = {}
        env.reward_sigmas = {}

    env.reward_sigmas_ema[key] = (
        sigma_update_rate * env.reward_sigmas_ema.get(key, torch.tensor([100.], device=env.device)) + (1 - sigma_update_rate) * error
    )
    env.reward_sigmas[key] = torch.minimum(env.reward_sigmas_ema[key], env.reward_sigmas.get(key, torch.tensor([100.], device=env.device))).clip(min=1e-8)
    return torch.sqrt(env.reward_sigmas[key])


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float | str) -> torch.Tensor:
    """Exponential reward on global anchor position tracking error."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
        env: ManagerBasedRLEnv, command_name: str, std: float | str) -> torch.Tensor:
    """Exponential reward on global anchor orientation tracking error."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str, body_names: list[str] | None = None
) -> torch.Tensor:
    """Exponential reward on relative body position tracking error (averaged)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    ).mean(dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str, body_names: list[str] | None = None
) -> torch.Tensor:
    """Exponential reward on relative body orientation tracking error (averaged)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    ).mean(dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str, body_names: list[str] | None = None
) -> torch.Tensor:
    """Exponential reward on global body linear velocity tracking error (averaged)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    ).mean(dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float | str, body_names: list[str] | None = None
) -> torch.Tensor:
    """Exponential reward on global body angular velocity tracking error (averaged)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    ).mean(dim=-1)
    std = _get_adaptive_sigma(env, std, error.mean())
    return torch.exp(-error / std**2)


def feet_stance_time(
        env: ManagerBasedRLEnv, asset_name: str, feet_names: list[str], vel_threshold: float, desired_time: float
) -> torch.Tensor:
    """Reward that prefers feet to maintain stance for desired durations.

    Tracks per-foot stance durations (zero crossing reset when a foot leaves
    stance) and penalizes if stance time falls below target. Useful to induce
    clean biped stepping with reduced scuffing on rough ground.
    """
    if not hasattr(env, '_buf_feet_stance_time'):
        env._buf_feet_stance_time = torch.zeros(env.num_envs, 2, device=env.device)

    robot = env.scene.articulations[asset_name]
    feet_indexes = [robot.body_names.index(name) for name in feet_names]

    stance = robot.data.body_link_lin_vel_w[:, feet_indexes].norm(dim=-1) < vel_threshold

    first_slide = (env._buf_feet_stance_time > 0.) * (~stance)
    rew_stanceTime = torch.sum((env._buf_feet_stance_time - desired_time).clip(max=0.) * first_slide, dim=1)

    env._buf_feet_stance_time += env.step_dt
    env._buf_feet_stance_time *= stance
    return rew_stanceTime


def swing_foot_height_bonus(
        env: ManagerBasedRLEnv,
        height_margin: float,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward lifting swing feet higher than base by a margin.

    For each foot currently in air (based on contact sensor's current_air_time > 0),
    adds a bonus proportional to ReLU(foot_z - base_z - height_margin).
    """
    # sensors and asset
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    # indices
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = list(range(len(asset.body_names)))
    # states
    base_z = asset.data.root_pos_w[:, 2]
    foot_z = asset.data.body_pos_w[:, body_ids, 2]
    # air mask
    air_time = contact_sensor.data.current_air_time[:, body_ids]
    in_air = air_time > 0.0
    # bonus
    bonus = (foot_z - base_z.unsqueeze(-1) - height_margin).clip(min=0.0)
    bonus = (bonus * in_air.float()).sum(dim=1)
    return bonus


def double_support_penalty(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        command_name: str,
        min_speed: float
) -> torch.Tensor:
    """Penalize double support when commanded speed is non-trivial.

    If both feet are in contact and the command XY speed exceeds `min_speed`,
    returns 1.0 per env; otherwise 0.0. Use negative weight in config.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # contact mask: current_contact_time > 0 means in contact
    contact = contact_sensor.data.current_contact_time
    # both feet in contact
    both_contact = (contact > 0.0).sum(dim=1) >= 2
    # command speed
    cmd = env.command_manager.get_command(command_name)
    cmd_speed = torch.linalg.norm(cmd[:, :2], dim=1)
    return (both_contact & (cmd_speed >= min_speed)).float()


def feet_gait(
        env: ManagerBasedRLEnv,
        period: float,
        offset: list[float],
        sensor_cfg: SceneEntityCfg,
        threshold: float = 0.5,
        command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward





def leg_joint_vel_symmetry_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    clip: float | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    # ---------- cache joint index pairs ----------
    if not hasattr(env, "_leg_vel_symmetry_cache"):
        # 1. 用 regex 找到所有目标 joint indices
        joint_ids, joint_names = asset.find_joints(asset_cfg.joint_names)

        # 2. 按名字分左右腿（假设 L_/R_ 命名）
        left = {}
        right = {}

        for jid, name in zip(joint_ids, joint_names):
            if name.startswith("Left_"):
                left[name[5:]] = jid
            elif name.startswith("Right_"):
                right[name[6:]] = jid

        # 3. 取交集，形成 (L, R) index 对
        pairs = []
        for k in left.keys() & right.keys():
            pairs.append((left[k], right[k]))

        assert len(pairs) > 0, \
            f"No left-right joint pairs found for {asset_cfg.joint_names}"

        env._leg_vel_symmetry_cache = pairs

    pairs = env._leg_vel_symmetry_cache

    # ---------- compute reward ----------
    vel = asset.data.joint_vel  # [num_envs, num_joints]
    reward = torch.zeros(env.num_envs, device=env.device)

    for l_id, r_id in pairs:
        diff = vel[:, l_id] - vel[:, r_id]
        reward += diff * diff

    reward /= len(pairs)

    if clip is not None:
        reward = torch.clamp(reward, max=clip)

    return reward