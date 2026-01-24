FROM osrf/ros:humble-desktop-full
LABEL maintainer="杨涌玉"
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-pip \
    ros-humble-rosidl-default-generators \
    ros-humble-webots-ros2 \
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
    tmux && \
    vim && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install pybind11 pybind11-stubgen
RUN mkdir -p /WarriorBot_dev
WORKDIR /WarriorBot_dev
COPY Booster_SDK ./Booster_SDK
COPY webots /usr/local/webots
COPY webots_simulation ./webots_simulation
RUN mkdir -p ros2_ws/src
ENV WEBOTS_HOME=/usr/local/webots
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WEBOTS_HOME/lib/controller
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN cd Booster_SDK && \
       ./install.sh && \
       mkdir build && \
       cd build && \
       cmake .. -DBUILD_PYTHON_BINDING=on && \
       make && \
       make install	
