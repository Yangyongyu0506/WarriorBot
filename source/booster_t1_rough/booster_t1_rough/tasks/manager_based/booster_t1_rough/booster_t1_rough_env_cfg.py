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
from .mdp.rewards import feet_gait, leg_joint_vel_symmetry_l2
# Pre-defined configs
from .robots.booster import BOOSTER_T1_CFG  # isort: skip

from .mdp.rewards import *
from .mdp.terminations import *


#### 定义爬行时的理想参考姿态（让四肢像四足动物一样分布）
CRAWLING_POSE = {
    # --- 手臂：强制向胸口内收并撑地 ---
    "Left_Shoulder_Pitch": 0.8,    # 向前摆动
    "Right_Shoulder_Pitch": 0.8,
    "Left_Shoulder_Roll": 1.3,      # 关键：根据URDF轴向，1.3弧度能将胳膊从侧向转为指向地面
    "Right_Shoulder_Roll": -1.3,    # 关键：对称内收
    "Left_Elbow_Pitch": 1.8,        # 肘部大幅弯曲，形成支撑
    "Right_Elbow_Pitch": 1.8,
    "Left_Elbow_Yaw": 0.0,
    "Right_Elbow_Yaw": 0.0,
    
    # --- 腿部：深度深蹲跪姿 ---
    "Left_Hip_Pitch": -1.2,         # 大腿向前收
    "Right_Hip_Pitch": -1.2,
    "Left_Hip_Roll": 0.0,           # 保持腿部平行
    "Right_Hip_Roll": 0.0,
    "Left_Knee_Pitch": 2.1,         # 膝盖近乎折叠 (限位2.18)
    "Right_Knee_Pitch": 2.1,
    "Left_Ankle_Pitch": -0.5,
    "Right_Ankle_Pitch": -0.5,
    
    "Waist": 0.0,
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.4,              # 稍微抬头，防止相机/传感器直接撞地
}



# 必须完全匹配 URDF 中的 link name
EXTREMITIES_NAME = ["left_hand_link", "right_hand_link", "left_foot_link", "right_foot_link"]


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.7, use_default_offset=True)


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False, # 设为 False，因为 heading 逻辑默认绑定在 X 轴，会干扰 Z 轴爬行
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # 我们在这里使用 lin_vel_x 这个插槽来代表“躯干方向的前进速度”
            lin_vel_x=(0.3, 0.8),    # 这里的 x 逻辑上对应躯干方向 (Body Z)
            lin_vel_y=(-0.2, 0.2),   # 侧移
            ang_vel_z=(-0.5, 0.5)    # 转向
        ),
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 1. 告诉策略：你看到的第一个线速度分量是 Body Z
        base_lin_vel = ObsTerm(func=base_lin_vel_crawling) 
        base_ang_vel = ObsTerm(func=base_ang_vel_crawling)
        projected_gravity = ObsTerm(func=projected_gravity_crawling)
        
        # 2. 这里不需要专门的 crawling_velocity_commands 了，
        # 因为我们已经在上面的 base_lin_vel 中把 Z 挪到了第一个位置，
        # 刚好对应 Commands 中的 lin_vel_x (Index 0)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        # Critic 也需要同步修改
        base_lin_vel = ObsTerm(func=base_lin_vel_crawling)
        base_ang_vel = ObsTerm(func=base_ang_vel_crawling)
        projected_gravity = ObsTerm(func=projected_gravity_crawling)
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

    # 1. 接触检查（这个 mdp.illegal_contact 在 locomotion.mdp 里有，可以保留）
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["H1", "H2"]), 
    #         "threshold": 1.0
    #     },
    # )

    # 2. 翻转检查
    # 修改：直接使用本地函数 bad_orientation，去掉 mdp. 前缀
    # bad_orientation = DoneTerm(
    #     func=bad_orientation, 
    #     params={
    #         "limit_angle": 1.0,
    #         "asset_cfg": SceneEntityCfg("robot")
    #     },
    # )

    # 3. 高度检查
    # 修改：直接使用本地函数 root_height_below，去掉 mdp. 前缀
    # root_height_below = DoneTerm(
    #     func=root_height_below,
    #     params={"threshold": 0.08, "asset_cfg": SceneEntityCfg("robot")},
    # )




@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

@configclass
class T1Rewards:
    """Reward terms for Booster T1 rough terrain locomotion."""
    
    # 核心驱动力：速度跟踪（command ≠ 0 时，不动就是负反馈）
    # 修改后的爬行线速度跟踪
    track_lin_vel_yz = RewTerm(
        func=track_lin_vel_yz_body_exp, # 使用你刚刚定义的自定义函数
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # 修改后的爬行角速度跟踪
    track_ang_vel_z = RewTerm(
        func=track_ang_vel_x_body_exp, # 使用新定义的函数
        weight=1.2,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 添加：躯干接触地面惩罚（不终止，但扣分）
    trunk_contact_penalty = RewTerm(
        func=mdp.contact_forces,
        weight=-0.1, # 负权重
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Trunk"),
            "threshold": 1.0,
        },
    )

    # 姿态奖励（防止翻滚或抬起头）
    base_orientation = RewTerm(
        func=crawling_orientation_l2, # 使用刚刚定义的函数
        weight=-3.0,   # 爬行时此权重依然要高，确保背部水平
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )



    # 动态稳定性惩罚：站立时不惩罚，趴下后才要求稳
    base_ang_vel = RewTerm(
        func=crawling_gated_ang_vel_yz_l2, 
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    base_lin_vel = RewTerm(
        func=crawling_gated_lin_vel_x_l2, 
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


    

    # 动态关节偏离惩罚：智能切换站立/爬行参考点
    # 这会强制机器人保持 CRAWLING_POSE 里的收拢姿态
    joint_deviation_dynamic = RewTerm(
        func=crawling_gated_joint_deviation,
        weight=-2.0, 
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "crawling_pose_dict": CRAWLING_POSE,
        },
    )

    # 关节偏离惩罚（保留腰 + 脖子）
    # 腰 + 脖子：极强
    torso_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "Waist",
                    "AAHead_yaw",
                    "Head_pitch",
                ],
            )
        },
    )

    # 2. 增加手臂外展惩罚 (防止 T-pose)
    arm_flaring_penalty = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_Shoulder_Roll"],
            )
        },
    )

    # 3. 增加手部高度惩罚 (强制手部贴近地面)
    hands_at_ground = RewTerm(
        func=crawling_hands_height_penalty,
        weight=-5.0, # 给一个明显的权重
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_hand_link"])
        },
    )



    
    # --- 摆动奖励：鼓励四肢迈步 ---
    # 爬行时，不仅是胯部，肩部也需要摆动来完成迈步
    limb_swing = RewTerm(
        func=mdp.joint_vel_l2,
        weight=0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", 
                joint_names=[".*_Hip_Pitch", ".*_Shoulder_Pitch"]
            )
        }
    )

    # --- 能量 & 平滑：确保动作自然且保护电机 ---
    joint_torque_penalty = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-5, # 保持较小，允许爬行起步时的大扭矩
    )
    action_rate_penalty = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )
    
    # --- 步态质量：四肢协同核心 ---

    # 滞空奖励：现在涵盖手和脚，鼓励四肢提速迈步
    foot_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.2,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=EXTREMITIES_NAME),
            "threshold": 0.4,
        },
    )

    # 四足对角爬行步态奖励
    # EXTREMITIES_NAME 顺序: [left_hand, right_hand, left_foot, right_foot]
    # offset [0.0, 0.5, 0.5, 0.0] 意味着:
    # - 左手(0.0) 和 右脚(0.0) 同步迈步
    # - 右手(0.5) 和 左脚(0.5) 同步迈步 (对角线步态)
    gait_phase = RewTerm(
        func=feet_gait,
        weight=1.0,
        params={
            "period": 0.8, # 爬行周期稍长，增加稳定性
            "offset": [0.0, 0.5, 0.5, 0.0], 
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=EXTREMITIES_NAME),
        },
    )

    # 左右不对称速度惩罚：防止跛行，涵盖肩部和胯部
    leg_vel_symmetry = RewTerm(
        func=leg_joint_vel_symmetry_l2,
        weight=-0.2, 
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Shoulder_.*"]),
            "clip": 5.0, 
        },
    )          

    # 足端/手端滑动惩罚
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=EXTREMITIES_NAME),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=EXTREMITIES_NAME),
        },
    )

    # --- 存活与基础奖励 ---
    # 调高存活奖励，鼓励机器人在艰难的“站立转爬行”初期多在场上探索
    alive = RewTerm(func=mdp.is_alive, weight=1.0) 

@configclass
class BoosterT1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: ActionsCfg = ActionsCfg()
    rewards: T1Rewards = T1Rewards()
    terminations: TerminationsCfg = TerminationsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    curriculum : CurriculumCfg = CurriculumCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
    
        # 核心修正：绕 Y 轴旋转 +90 度 (四元数: w=0.7071, y=0.7071)
        # 这样机器人的局部 X 轴就变成了指向头部的方向，局部 Z 轴变成了指向背部的方向
        crawling_quat = (0.7071, 0.0, 0.7071, 0.0) 
        self.scene.robot = BOOSTER_T1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.25),      # 趴下后的高度
                rot=crawling_quat,         # 让 X 轴指向头顶
                joint_pos=CRAWLING_POSE,
                joint_vel={".*": 0.0},
            )
        )

        # 修正 Reset 事件：强制锁定 Pitch 为 1.57 (90度)
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-3.14, 3.14),
                "pitch": (1.57, 1.57),  # 对应 90 度
                "roll": (0.0, 0.0),
            },
            "velocity_range": {"x": (0,0), "y": (0,0), "z": (0,0), "roll": (0,0), "pitch": (0,0), "yaw": (0,0)},
        }

        # 因为 X 轴已经是头了，我们直接使用系统标准观察项，不需要自定义映射了！
        self.observations.policy.base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        self.observations.policy.base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        self.observations.policy.projected_gravity = ObsTerm(func=mdp.projected_gravity)

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