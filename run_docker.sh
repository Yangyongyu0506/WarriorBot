xhost +local:root
docker run \
    --gpus all \
    -it \
    -d \
    --network host \
    --privileged \
    --runtime=nvidia \
    -e DISPLAY=$DISPLAY \
    -v ./ros2_ws:/WarriorBot_dev/ros2_ws \
    -v ./scripts:/WarriorBot_dev/scripts \
    -v ./booster_assets:/WarriorBot_dev/booster_assets \
    -v /dev:/dev \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -e "PRIVACY_CONSENT=Y" \
    -e "ACCEPT_EULA=Y" \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    --name warriorsim \
    warriorsim:latest
docker exec -it warriorsim bash -c "tmux"
docker rm -f warriorsim