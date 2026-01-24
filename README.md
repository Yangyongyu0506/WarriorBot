# WarriorBot
This repository is for caching code for the Warrior Competition.
## 教程
### 加速进化机器人说明书
参见[https://booster.feishu.cn/wiki/H2Dowdnokij7p8ks9K3cZPuJnOg](url)
### 应用开发仓库
参见[https://github.com/BoosterRobotics/booster_robotics_sdk.git](url)或[https://gitee.com/booster_robotics/sdk_release.git](url)
## 仿真部署
### 环境配置（NIS）
仿真软件选择Isaac Sim，使用Docker管理环境。参见[https://github.com/arambarricalvoj/nvidia_isaac-sim_ros2_docker.git](url)

本人API Key：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

若要拉取Nvidia的Isaac Sim镜像（NIS），必须先注册NGC账号并获取API Key，参见[https://docs.nvidia.com/ngc/latest/ngc-user-guide.html#generating-api-key](url)。然后使用以下命令登录：
```bash
docker login nvcr.io --username '$oauthtoken' --password '<your API Key>'
```
这个地方我死活拉去不下来NIS镜像，因为它实在是太大了，所以我先使用webots仿真。
### 环境配置（Webots）
首先安装从Booster文档中把webots软件和仿真配置都下载下来解压。然后配置一个带有webots和ros-humble的镜像。
