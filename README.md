# WarriorBot
This repository is for caching code for the Warrior Competition.
## 教程
### 加速进化机器人说明书
参见[https://booster.feishu.cn/wiki/H2Dowdnokij7p8ks9K3cZPuJnOg](url)
### 应用开发仓库
参见[https://github.com/BoosterRobotics/booster_robotics_sdk.git](url)或[https://gitee.com/booster_robotics/sdk_release.git](url)
## 仿真部署
### 环境配置（NIS）
仿真软件选择Isaac Sim，使用Docker管理环境。

若要拉取Nvidia的Isaac Sim镜像（NIS），必须先注册NGC账号并获取API Key，参见[https://docs.nvidia.com/ngc/latest/ngc-user-guide.html#generating-api-key](url)。然后使用以下命令登录：
```bash
docker login nvcr.io --username '$oauthtoken' --password '<your API Key>'
```
在isaac sim的容器里运行一键启动脚本失败了，因为该脚本假设isaac sim和ros2在同一环境。并且该脚本是makeself脚本，内容极大，无法用文本编辑器查看。运行：
```bash
sh isaac_package_t1_7dof_arms_hand_0.0.2.run --noexec --keep
```
首先制作一个同时带isaac sim 4.2.0和ros-humble的镜像（见Dockerfile），然后运行run_docker.sh启动镜像即可。
## Dockerfile使用方法
需要将Booster_SDK和booster_assets文件夹下载到WarriorBot目录下，以供镜像内的环境配置。其次在scripts文件夹下将[https://booster.feishu.cn/wiki/H2Dowdnokij7p8ks9K3cZPuJnOg](url)中的isaac_package_0.0.7.run脚本和booster-runner-full-0.0.11.run脚本下载到其中。由于这两个脚本需要在容器中运行，所以需要运行上述指令解压并进行一定程度修改。
### 脚本修改
在scripts目录下运行：
```bash
sh isaac_package_0.0.7.run --noexec --keep
sh booster-runner-full-0.0.11.run --noexec --keep
```
将两脚本解压。随后在isaac_package_0.0.7.run生成的目录中修改.sh脚本，将其中的default_isaac_path变量修改为/isaac-sim/python.sh，在booster-runner-full-0.0.11.run生成的目录中修改.sh脚本将所有命令前的sudo去掉（因为在容器内运行本身就是root权限）。之后运行两个目录下的readme.md将目录压缩回脚本即可。

### 构建脚本及运行
完成以上工作后，在WarriorBot目录下运行：
```bash
docker build . -t warriorsim:latest # 由于isaac-sim4.2.0的镜像非常大，因此需要稳定的网络环境，并且时间会很长
bash run_docker.sh # 运行此脚本来启动容器
```
### 注意事项
- 如非必要，不要改变目录结构，否则Docker镜像构建会失败。
- 本教程虽然只提到了4自由度手机器人脚本的修改方法，但是其他模型机器人脚本修改方法一样。
- 制备同时带有isaac-sim和ros-humble的镜像，参见[https://github.com/arambarricalvoj/nvidia_isaac-sim_ros2_docker.git](url)
- 只有灵巧手模型的机器人仿真会发布/tf话题，这是因为只有灵巧手机器人的脚本中启动了robot_state_publisher节点。其他脚本经过类似的修改也可以实现相同效果。
### 4dof手臂模型机器人的isaac-sim仿真如何调出/tf话题
修改isaac_package/start_ros2_local_isaac_sim.sh。首先要加入robot_state_publisher节点，然后作话题重映射：
```bash
# start robot_state_publisher
killall robot_state_publisher
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="$(xacro /WarriorBot_dev/booster_assets/robots/T1/T1_23dof.urdf)" \
  -r /joint_states:=/booster/ros2_k2_joint_states &
```
## ROS2开发
所有比赛用代码均源于ros2_ws/src/warriordev包。

### setup.py的修改
为了使得colcon能够一并将launch文件和配置文件打包构建，需要在setup.py中加如下代码：
```python
import os
from glob import glob
...
        (f'share/{package_name}/config', glob('config/*')), # include configurations
        (f'share/{package_name}/launch', glob('launch/*')), # include launch files
...
```
### ROS2节点的安全退出
由于大多数机器人都是典型的大小脑架构，在上位机退出程序时并不会自动发送指令终止下位机，因此如果机器人正在行走，我们把上层操作节点杀掉，机器人就会保持原速行走，非常危险。

我们在ctrl+C中断程序的时候，rclc底层的执行器会在Python的try-except块捕捉到错误前出错，因此必须引入rclpy.executors的ExternalShutdownException。main函数按照以下框架写：
```python
def main():
    rclpy.init()
    node = SquareRouteNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        node.send_cmd(0.0, 0.0) # 这里停止下位机
        node.get_logger().warn("Shutting down SquareRouteNode...")
    finally:
        if rclpy.ok(): # 检测ros2的环境是否还在 
            node.destroy_node()
            rclpy.shutdown()
```