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
运行：
```bash
xhost +local:root && docker compose up -d
```
在isaac sim的容器里运行一键启动脚本失败了，因为该脚本假设isaac sim和ros2在同一环境。并且该脚本是makeself脚本，内容极大，无法用文本编辑器查看。运行：
```bash
sh isaac_package_0.0.7.run --noexec --keep
```
解压。接下来实现isaac sim启动与ros2启动解耦。

isaac sim自带一个最小的ROS2内核，我们需要指定一些环境变量才能使isaac sim发布ROS2话题：
```bash
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/isaac-sim/exts/isaacsim.ros2.bridge/humble/lib
export OMNI_KIT_ALLOW_ROOT=1
```

由于isaac sim自带一个特殊的python环境，所以启动仿真脚本必须输入如下命令：
```bash
/isaac-sim/python.sh \
  /isaac-sim/booster_sim/booster_isaac/booster_standalone_ros2_robocup_t1.py
```