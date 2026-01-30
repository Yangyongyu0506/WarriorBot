from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task

from .loco_rough import T1WalkControllerCfg

"""这个任务专门适配基于booster_t1_rough强化学习的部署任务"""

@configclass
class T1RoughControllerCfg(T1WalkControllerCfg):
    '''Rough-terrain walk for T1 robot.'''
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "models/policy.pt"

register_task(
    "t1_rough", T1RoughControllerCfg())