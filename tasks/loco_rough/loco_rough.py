from __future__ import annotations
from dataclasses import MISSING
import os
import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import (
    ControllerCfg, PolicyCfg, VelocityCommandCfg
)
from booster_deploy.robots.booster import T1_23DOF_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.isaaclab import math as lab_math


class LocomotionPolicy(Policy):
    """
    IsaacLab-style locomotion policy.

    Conventions:
    - policy order: joint order used by the neural network
    - sim order:    joint order used by simulator / real robot
    """

    def __init__(self, cfg: LocomotionPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot

        # --------------------------------------------------
        # Load TorchScript policy
        # --------------------------------------------------
        policy_path = cfg.checkpoint_path
        if not os.path.isabs(policy_path):
            policy_path = os.path.join(self.task_path, policy_path)

        self._model: torch.jit.ScriptModule = torch.jit.load(
            policy_path, map_location="cpu"
        ).eval()

        # --------------------------------------------------
        # Joint order mappings
        # --------------------------------------------------
        self.policy_joint_names = cfg.policy_joint_names
        self.sim_joint_names = self.robot.cfg.joint_names

        self.policy_to_sim = torch.tensor(
            [self.sim_joint_names.index(name)
             for name in self.policy_joint_names],
            dtype=torch.long,
        )
        self.sim_to_policy = torch.argsort(self.policy_to_sim)

        # --------------------------------------------------
        # Policy state
        # --------------------------------------------------
        self.last_action = torch.zeros(
            len(self.policy_joint_names),
            dtype=torch.float32,
        )

    def reset(self) -> None:
        self.last_action.zero_()

    # --------------------------------------------------
    # Observation
    # --------------------------------------------------
    def compute_observation(self) -> torch.Tensor:
        """
        Construct observation tensor in POLICY JOINT ORDER.
        Must exactly match IsaacLab training configuration.
        """

        # --- robot state (sim order) ---
        joint_pos_sim = self.robot.data.joint_pos
        joint_vel_sim = self.robot.data.joint_vel

        # --- reorder to policy order ---
        joint_pos = joint_pos_sim[self.policy_to_sim]
        joint_vel = joint_vel_sim[self.policy_to_sim]
        default_pos = self.robot.default_joint_pos[self.policy_to_sim]

        # --- base state ---
        base_quat = self.robot.data.root_quat_w
        base_ang_vel = self.robot.data.root_ang_vel_b

        gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        projected_gravity = lab_math.quat_apply_inverse(base_quat, gravity_w)

        # --- velocity command ---
        cmd = torch.tensor(
            [
                self.controller.vel_command.lin_vel_x,
                self.controller.vel_command.lin_vel_y,
                self.controller.vel_command.ang_vel_yaw,
            ],
            dtype=torch.float32,
        )

        # --- observation vector (policy order) ---
        obs = torch.cat(
            [
                base_ang_vel,
                projected_gravity,
                cmd,
                joint_pos - default_pos,
                joint_vel * self.cfg.obs_dof_vel_scale,
                self.last_action,
            ],
            dim=0,
        )

        return obs

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    def inference(self) -> torch.Tensor:
        """
        Run policy inference.

        Returns:
            q_target_sim (torch.Tensor):
                Joint position targets in SIM JOINT ORDER.
        """

        # 1. observation (policy order)
        obs = self.compute_observation()

        # 2. network forward
        with torch.no_grad():
            action = self._model(obs).squeeze(0)
            action = torch.clamp(action, -1.0, 1.0)

        # 3. update policy memory
        self.last_action.copy_(action)

        # 4. action -> joint target (policy order)
        default_pos_policy = self.robot.default_joint_pos[self.policy_to_sim]
        q_target_policy = default_pos_policy + self.cfg.action_scale * action

        # 5. map back to sim order
        q_target_sim = torch.empty_like(self.robot.default_joint_pos)
        q_target_sim[self.policy_to_sim] = q_target_policy

        return q_target_sim


@configclass
class LocomotionPolicyCfg(PolicyCfg):
    constructor = LocomotionPolicy
    checkpoint_path: str = MISSING  # type: ignore
    obs_dof_vel_scale: float = 1.0
    action_scale: float = 0.7
    policy_joint_names: list[str] = MISSING  # type: ignore
    enable_safety_fallback: bool = True


@configclass
class T1WalkControllerCfg(ControllerCfg):
    robot = T1_23DOF_CFG.replace(  # type: ignore
        default_joint_pos=[
            0, 0,
            0.2, -1.3, 0, -0.5,
            0.2,  1.3, 0,  0.5,
            0.,
            -0.2, 0, 0, 0.4, -0.2, 0.,
            -0.2, 0, 0, 0.4, -0.2, 0.
        ],
        joint_stiffness=[
            10.0, 10.0,
            50.0, 50.0, 50.0, 50.0,
            50.0, 50.0, 50.0, 50.0,
            200.,
            200.0, 200.0, 200.0, 200.0, 50.0, 50.0,
            200.0, 200.0, 200.0, 200.0, 50.0, 50.0,
        ],
        joint_damping=[
            1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            5.0,
            5.0, 5.0, 5.0, 5.0, 1.0, 1.0,
            5.0, 5.0, 5.0, 5.0, 1.0, 1.0,
        ],
    )

    vel_command: VelocityCommandCfg = VelocityCommandCfg(
        vx_max=1.0,
        vy_max=1.0,
        vyaw_max=1.0,
    )

    policy: LocomotionPolicyCfg = LocomotionPolicyCfg(
        obs_dof_vel_scale=1.0,
        policy_joint_names=[
            'AAHead_yaw', 
            'Left_Shoulder_Pitch', 
            'Right_Shoulder_Pitch', 
            'Waist', 
            'Head_pitch', 
            'Left_Shoulder_Roll', 
            'Right_Shoulder_Roll', 
            'Left_Hip_Pitch', 
            'Right_Hip_Pitch', 
            'Left_Elbow_Pitch', 
            'Right_Elbow_Pitch', 
            'Left_Hip_Roll', 
            'Right_Hip_Roll', 
            'Left_Elbow_Yaw', 
            'Right_Elbow_Yaw', 
            'Left_Hip_Yaw', 
            'Right_Hip_Yaw', 
            'Left_Knee_Pitch', 
            'Right_Knee_Pitch', 
            'Left_Ankle_Pitch', 
            'Right_Ankle_Pitch', 
            'Left_Ankle_Roll', 
            'Right_Ankle_Roll'
        ],
    )
