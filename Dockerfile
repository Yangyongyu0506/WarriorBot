# Base image
FROM nvcr.io/nvidia/isaac-sim:4.2.0
LABEL maintainer="杨涌玉"

ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------
# Install ROS2 Humble (offline-key / mirror friendly)
# ------------------------------

RUN apt-get update && apt-get install -y locales \
 && locale-gen en_US en_US.UTF-8 \
 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    gnupg2 \
    lsb-release

RUN add-apt-repository universe

# Copy pre-downloaded ROS GPG key (avoid GitHub access during build)
COPY ros.key /usr/share/keyrings/ros-archive-keyring.gpg

# Add ROS2 apt repository
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
 https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu $(lsb_release -cs) main" \
 > /etc/apt/sources.list.d/ros2.list

# Install ROS2 Humble
RUN apt-get update \
 && apt-get install -y ros-humble-desktop \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
# ------------------------------

RUN apt-get update && apt-get install -y \
    cmake \
    python3-colcon-common-extensions \
    python3-pip \
    ros-humble-rosidl-default-generators \
    ros-humble-ament-cmake \
    libasio-dev \
    libtinyxml2-dev \
    wget \
    ninja-build \
    libgtest-dev \
    libgoogle-glog-dev \
    libboost-dev \
    libeigen3-dev \
    liblua5.3-dev \
    graphviz \
    libgraphviz-dev \
    libcurl4-openssl-dev \
    libsdl2-dev \
    joystick \
    libspdlog-dev \
    tmux \
    vim && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install pybind11 pybind11-stubgen

# Install ROS2 bridge for Isaac Sim
RUN apt-get update && apt-get install -y \
    ros-humble-ros-ign-bridge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV ROS_DISTRO=humble
ENV ROS_VERSION=2
ENV ROS_PYTHON_VERSION=3
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV OMNI_KIT_ALLOW_ROOT=1

RUN mkdir -p /WarriorBot_dev
WORKDIR /WarriorBot_dev
COPY Booster_SDK ./Booster_SDK
COPY booster_assets ./booster_assets
COPY fastdds_profile.xml ./fastdds_profile.xml
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/WarriorBot_dev/fastdds_profile.xml
RUN mkdir scripts
RUN mkdir -p ros2_ws/src
COPY scripts/ ./scripts/
COPY ros2_ws/src/ ./ros2_ws/src/

RUN cd /WarriorBot_dev/Booster_SDK && \
    ./install.sh && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_PYTHON_BINDING=on && \
    make && \
    make install

# Source ROS2
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc