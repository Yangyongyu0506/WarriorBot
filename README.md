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
sh isaac_package_t1_7dof_arms_hand_0.0.2.run --noexec --keep
```
首先制作一个同时带isaac sim 4.2.0和ros-humble的镜像（见Dockerfile），然后运行run_docker.sh启动镜像即可。
## Dockerfile使用方法
需要将Booster_SDK和booster_assets文件夹下载到WarriorBot目录下，以供镜像内的环境配置。其次在scripts文件夹下将[https://booster.feishu.cn/wiki/H2Dowdnokij7p8ks9K3cZPuJnOg](url)中的isaac_package_0.0.7.run脚本和booster-runner-full-0.0.11.run脚本下载到其中。由于这两个脚本需要在容器中运行，所以需要运行上述指令解压并进行一定程度修改。
#### 脚本修改
在scripts目录下运行：
```bash
sh isaac_package_0.0.7.run --noexec --keep
sh booster-runner-full-0.0.11.run --noexec --keep
```
将两脚本解压。随后在isaac_package_0.0.7.run生成的目录中修改.sh脚本，将其中的default_isaac_path变量修改为/isaac-sim/python.sh，在booster-runner-full-0.0.11.run生成的目录中修改.sh脚本将所有命令前的sudo去掉（因为在容器内运行本身就是root权限）。之后运行两个目录下的readme.md将目录压缩回脚本即可。

最后直接运行run_docker.sh即可启动docker容器，其名为warriorsim。