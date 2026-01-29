# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for Booster T1 rough-terrain walking.

Sets runner, policy, and algorithm hyperparameters. Tune `num_steps_per_env`,
`max_iterations`, and learning rate schedule per terrain curriculum stage.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
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
    )