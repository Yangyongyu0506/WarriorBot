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





################################## 特定于 Booster T1 Claw Rough 任务的奖励项 #####################################
def track_lin_vel_yz_body_exp(
    env: "ManagerBasedRLEnv", 
    std: float, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """奖励机器人本体坐标系下的 Z 速度跟踪指令 X，Y 速度跟踪指令 Y。
    
    适用于爬行模式：
    - 指令 X (前进) -> 对应机器人本地 Z 轴
    - 指令 Y (侧移) -> 对应机器人本地 Y 轴
    """
    # 1. 获取指令 (通常是 [vx, vy, wz])，提取前两个元素 [vx, vy]
    command = env.command_manager.get_command(command_name)[:, :2]
    
    # 2. 获取机器人资产
    asset = env.scene[asset_cfg.name]
    
    # 3. 获取本体坐标系下的线速度 (root_lin_vel_b 包含 [vx, vy, vz])
    # 在爬行位姿下：
    # vel_b[:, 0] 是原胸口方向 (现在指向地面)
    # vel_b[:, 1] 是原左侧方向 (现在依然是侧向)
    # vel_b[:, 2] 是原头顶方向 (现在指向前方)
    vel_b = asset.data.root_lin_vel_b
    
    # 4. 提取当前的 [前向, 侧向] 速度，即 [Body_Z, Body_Y]
    actual_vel_yz = vel_b[:, [2, 1]]
    
    # 5. 计算平方误差
    error = torch.sum(torch.square(command - actual_vel_yz), dim=1)
    
    # 6. 返回指数奖励
    return torch.exp(-error / (std**2))




def track_ang_vel_x_body_exp(
    env: "ManagerBasedRLEnv", 
    std: float, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # 1. 获取指令中的 wz (转向)
    command_wz = env.command_manager.get_command(command_name)[:, 2]
    
    # 2. 获取机器人资产
    asset = env.scene[asset_cfg.name]
    
    # 3. 获取本体坐标系下的角速度 [wx, wy, wz]
    ang_vel_b = asset.data.root_ang_vel_b
    
    # 4. 【关键修正】：由于本体 X 轴指向地面，
    # 世界坐标系的 +Yaw 对应本体坐标系的 -wx
    actual_yaw_vel = -ang_vel_b[:, 0]  # 注意这个负号
    
    # 5. 计算平方误差
    error = torch.square(command_wz - actual_yaw_vel)
    return torch.exp(-error / (std**2))


def crawling_orientation_l2(
    env: "ManagerBasedRLEnv", 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """惩罚身体姿态偏离'脸朝下'的水平爬行姿态。
    
    在理想爬行姿态下：
    - 重力向量 (0,0,-1) 在本体坐标系应投影为 (1, 0, 0)
    - 我们惩罚重力在本体 Y 和 Z 轴上的分量
    """
    # 获取重力投影 (n_envs, 3) -> [gx, gy, gz]
    projected_gravity = env.scene[asset_cfg.name].data.projected_gravity
    
    # 惩罚 gy 和 gz 的大小（即身体倾斜或翻滚）
    # projected_gravity[:, [1, 2]] 提取 Y 和 Z 分量
    return torch.sum(torch.square(projected_gravity[:, [1, 2]]), dim=1)



def crawling_gated_ang_vel_yz_l2(
    env: "ManagerBasedRLEnv", 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """门控角速度惩罚：只有当身体趋于水平时，才惩罚晃动。"""
    asset = env.scene[asset_cfg.name]
    # 获取重力在 Body X 上的投影 (站在 0 左右，趴在 1 左右)
    # 我们取 max(0, gx)，确保只有当身体向前倒时才开始计算惩罚
    gate = torch.clamp(asset.data.projected_gravity[:, 0], min=0.0)
    
    # 计算原本的惩罚
    ang_vel_b = asset.data.root_ang_vel_b
    penalty = torch.sum(torch.square(ang_vel_b[:, [1, 2]]), dim=1)
    
    # 门控：站立时 gate=0, 惩罚为0；趴下时 gate=1, 全额惩罚
    return gate * penalty

def crawling_gated_lin_vel_x_l2(
    env: "ManagerBasedRLEnv", 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """门控线速度惩罚：只有当身体趋于水平时，才惩罚垂直颠簸。"""
    asset = env.scene[asset_cfg.name]
    gate = torch.clamp(asset.data.projected_gravity[:, 0], min=0.0)
    
    lin_vel_b = asset.data.root_lin_vel_b
    penalty = torch.square(lin_vel_b[:, 0])
    
    return gate * penalty



#能够识别当前“姿态目标”的门控函数
def crawling_gated_joint_deviation(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    crawling_pose_dict: dict
) -> torch.Tensor:
    """动态关节偏离惩罚：在站立和爬行姿态之间平滑切换。
    
    Args:
        crawling_pose_dict: 爬行状态下的目标关节位置 (Dict[str, float])
    """
    asset = env.scene[asset_cfg.name]
    curr_joint_pos = asset.data.joint_pos
    default_stand_pos = asset.data.default_joint_pos # 机器人默认的站立姿态
    
    # 1. 构造爬行姿态的 Tensor (需要与当前关节顺序对应)
    # 这一步通常在初始化时做更好，但为了演示逻辑直接写在这里
    crawling_target = default_stand_pos.clone()
    joint_names = asset.joint_names
    for name, pos in crawling_pose_dict.items():
        if name in joint_names:
            idx = joint_names.index(name)
            crawling_target[:, idx] = pos

    # 2. 计算门控 (gate): gx = 0 (站立), gx = 1 (趴下)
    gate = torch.clamp(asset.data.projected_gravity[:, 0], min=0.0, max=1.0).unsqueeze(1)

    # 3. 计算当前目标 (在站立和爬行目标之间线性插值)
    # 站立时目标是 default_stand_pos，趴下时目标是 crawling_target
    dynamic_target = (1.0 - gate) * default_stand_pos + gate * crawling_target

    # 4. 计算 L1 偏离
    return torch.sum(torch.abs(curr_joint_pos - dynamic_target), dim=1)