# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for Booster T1 rough-terrain walking.

Sets runner, policy, and algorithm hyperparameters. Tune `num_steps_per_env`,
`max_iterations`, and learning rate schedule per terrain curriculum stage.
"""

import torch
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl.symmetry_cfg import RslRlSymmetryCfg

def _act_mirror_fn(act: torch.Tensor) -> torch.Tensor:
    booster_joint_mirror_map = [
        0, 2, 1, 3, 4,
        6, 5,
        8, 7, 10, 9,
        12, 11, 14, 13,
        16, 15, 18, 17,
        20, 19, 22, 21
    ]

    negate_mask = torch.tensor(
        [
            1, 1, 1, 1, 1,
            -1, -1,
            1, 1, 1, 1,
            -1, -1, -1, -1, -1, -1,
            1, 1, 1, 1,
            -1, -1,
        ],
        dtype=torch.float32,
        device=act.device,
    )

    act_m = act[:, booster_joint_mirror_map] * negate_mask

    return torch.nan_to_num(
        act_m, nan=0.0, posinf=0.0, neginf=0.0
    )   


def _obs_mirror_fn(obs_td):
    # 深拷贝 TensorDict + tensor
    obs_td_m = obs_td.clone(False)

    policy_obs = obs_td["policy"]
    policy_obs_m = policy_obs.clone()

    # base_ang_vel
    policy_obs_m[:, 0] *= -1
    policy_obs_m[:, 2] *= -1

    # projected gravity
    policy_obs_m[:, 4] *= -1

    # command
    policy_obs_m[:, 7] *= -1
    policy_obs_m[:, 8] *= -1

    # joint pos / vel
    policy_obs_m[:, 9:32]  = _act_mirror_fn(policy_obs[:, 9:32])
    policy_obs_m[:, 32:55] = _act_mirror_fn(policy_obs[:, 32:55])

    # ⚠️ 强烈建议 crawl 阶段不 mirror last_action
    # policy_obs_m[:, 55:78] = _act_mirror_fn(policy_obs[:, 55:78])

    # 数值防护（关键）
    policy_obs_m = torch.nan_to_num(
        policy_obs_m, nan=0.0, posinf=0.0, neginf=0.0
    )

    obs_td_m.set_("policy", policy_obs_m)

    return obs_td_m



def crawl_symmetry_augmentation(
    env,
    obs,
    actions,
):
    obs_mirror = None
    actions_mirror = None

    if obs is not None:
        obs_mirror = _obs_mirror_fn(obs)

    if actions is not None:
        actions_mirror = _act_mirror_fn(actions)

    return obs_mirror, actions_mirror


@configclass
class PPOSymmetryRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16  # PPO rollout length per env (increase for harder terrain)
    max_iterations = 150  # total training iterations
    save_interval = 50  # checkpoint frequency
    experiment_name = "cartpole_direct"  # experiment label in logs
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # initial Gaussian action noise std
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[32, 32],  # compact MLPs; scale if observations grow
        critic_hidden_dims=[32, 32],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,  # critic weight
        use_clipped_value_loss=True,
        clip_param=0.2,  # PPO clip range
        entropy_coef=0.005,  # exploration weight
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",  # lr schedule policy
        gamma=0.99,
        lam=0.95,  # GAE lambda
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            data_augmentation_func=crawl_symmetry_augmentation,
            mirror_loss_coeff=0.05,
        ),
    )