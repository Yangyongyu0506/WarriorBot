# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Environment configuration for Booster T1 humanoid walking on rough terrain.

Composes scene, sensors, MDP commands, observations, rewards, terminations,
and curriculum using Isaac Lab's manager-based style. Tuned for biped gait
stability with height scanning, contact sensing, and biped-specific rewards.
"""

import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from .mdp.events import sync_action_offsets_to_defaults
from .mdp.terminations import is_fallen
from .mdp.rewards import feet_gait, feet_lateral_separation_exp, leg_joint_vel_symmetry_l2, leg_joint_pos_symmetry_l2, double_support_penalty, feet_stance_time, swing_foot_height_bonus, support_force_balance
# Pre-defined configs
from .robots.booster import BOOSTER_T1_CFG  # isort: skip

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.7, use_default_offset=True)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # ensure zero actions hold the pose after reset
    sync_action_offsets = EventTerm(
        func=sync_action_offsets_to_defaults,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Trunk"), "threshold": 1.0},
    )
    fallen = DoneTerm(
        func=is_fallen,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Waist"),
            "foot_names": ".*_foot_link",
            "min_clearance": 0.18,
            "tilt_threshold": 0.9,
            "persist_steps": 3,
        },
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

@configclass
class RewardsCfg:
    """Rewards for Booster T1 biped locomotion (anti-crutch, anti-shuffle)."""

    # =========================================================
    # 1. 任务驱动力（不动一定是劣解）
    # =========================================================
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
        },
    )

    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
        },
    )

    # =========================================================
    # 2. Base 稳定（不能靠双脚赖地）
    # =========================================================
    base_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-2.0,   # 比你之前略降，避免“僵尸站立”
    )

    base_ang_vel = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.1,
    )

    base_lin_vel = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-1.5,
    )

    # =========================================================
    # 3. 上半身约束（治帕金森，但不绑死）
    # =========================================================
    torso_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.8,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["Waist", "AAHead_yaw", "Head_pitch"],
            )
        },
    )

    arm_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "Left_Shoulder_Pitch",
                    "Left_Shoulder_Roll",
                    "Left_Elbow_Pitch",
                    "Left_Elbow_Yaw",
                    "Right_Shoulder_Pitch",
                    "Right_Shoulder_Roll",
                    "Right_Elbow_Pitch",
                    "Right_Elbow_Yaw",
                ],
            )
        },
    )

    # 关键：直接抑制高速抖动（比 action_rate 更直接）
    joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.08,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "Left_Shoulder_.*",
                    "Left_Elbow_.*",
                    "Right_Shoulder_.*",
                    "Right_Elbow_.*",
                    ".*_Ankle_.*",
                ],
            )
        },
    )

    # =========================================================
    # 4. 腿部：不强对称，只防塌陷
    # =========================================================
    knee_ankle_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_Knee_.*",
                    ".*_Ankle_.*",
                ],
            )
        },
    )

    right_leg_incite = RewTerm(
        func=mdp.joint_vel_l2,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["Right_Knee_Pitch", "Right_Hip_Pitch"],
            )
        }
    )

    # 防止“并腿锁死”
    hip_roll_penalty = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.6,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_Hip_Roll.*"],
            )
        },
    )

    # =========================================================
    # 5. 关键：接触拓扑（真正解决拐杖解）
    # =========================================================

    # ---- (1) 反双支撑：走路时不准两脚都踩死 ----
    double_support = RewTerm(
        func=double_support_penalty,
        weight=-0.35,   # 非常关键，但不能太大
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_foot_link"
            ),
            "command_name": "base_velocity",
            "min_speed": 0.2,
        },
    )

    # ---- (2) stance 时间：不准点地腿 ----
    feet_stance_time = RewTerm(
        func=feet_stance_time,
        weight=-0.05,
        params={
            "asset_name": "robot",
            "feet_names": ["left_foot_link", "right_foot_link"],
            "vel_threshold": 0.08,
            "desired_time": 0.3,
        },
    )

    # ---- (3) swing 脚高度：不准假摆 ----
    swing_foot_height = RewTerm(
        func=swing_foot_height_bonus,
        weight=0.05,
        params={
            "height_margin": 0.1,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_foot_link"
            ),
        },
    )

    # ---- (4) 强制把脚分开，防止自干涉 ----
    feet_separation = RewTerm(
        func=feet_lateral_separation_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_foot_link", "right_foot_link"]
            ),
            "min_dist": 0.5,
            "std": 0.5,
        },
    )

    # ---- (5) 强制两腿受力对称 ----
    support_force_balance = RewTerm(
        func=support_force_balance,
        weight=-3.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_foot_link", "right_foot_link"]
            ),
        },
    )

    # =========================================================
    # 6. 能量 & 平滑（兜底）
    # =========================================================
    joint_torque_penalty = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-5,
    )

    action_rate_penalty = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    # =========================================================
    # 7. 其他
    # =========================================================
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=".*_foot_link"
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_foot_link"
            ),
        },
    )

    alive = RewTerm(
        func=mdp.is_alive,
        weight=0.01,
    )

@configclass
class BoosterT1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    curriculum : CurriculumCfg = CurriculumCfg()
    events: EventCfg = EventCfg()
    def __post_init__(self):
        super().__post_init__()
        # Scene
        self.scene.robot = BOOSTER_T1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Waist"
        
        # Disable over-randomization (early learning killer)
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None

        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        # Commands (NO SIDEWAYS WALKING INITIALLY)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # Termination tuning
        self.terminations.base_contact.params["sensor_cfg"].body_names = "H2"
        self.terminations.fallen.params.update(
            dict(
                min_clearance=0.18,
                tilt_threshold=0.9,
                persist_steps=3,
            )
        )

@configclass
class BoosterT1RoughEnvCfg_PLAY(BoosterT1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None