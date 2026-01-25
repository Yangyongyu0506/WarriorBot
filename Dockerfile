FROM osrf/ros:humble-desktop-full 
LABEL maintainer="杨涌玉"

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

RUN mkdir -p /WarriorBot_dev
WORKDIR /WarriorBot_dev

COPY Booster_SDK ./Booster_SDK
COPY booster_assets ./booster_assets
COPY fastdds_profile.xml ./fastdds_profile.xml

ENV FASTRTPS_DEFAULT_PROFILES_FILE=/WarriorBot_dev/fastdds_profile.xml

RUN mkdir scripts
RUN mkdir -p ros2_ws/src
COPY ros2_ws/src/ ./ros2_ws/src/
COPY ./isaac_package/booster_ros2 ./booster_ros2

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /WarriorBot_dev/booster_ros2/install/setup.bash" >> ~/.bashrc

RUN cd /WarriorBot_dev/Booster_SDK && \
    ./install.sh && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_PYTHON_BINDING=on && \
    make && \
    make install
