# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based task package for Booster T1 rough-terrain RL.

This package wires up Isaac Lab's manager-based environment style and
exposes humanoid locomotion tasks, configurations, and helpers. Importing
this module ensures gym registration side-effects and convenient access
to subpackages.
"""

import gymnasium as gym  # noqa: F401
