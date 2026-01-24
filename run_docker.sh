xhost +local:root
docker run -it \
	   --rm \
	   --gpus all \
	   --runtime=nvidia \
	   --network host \
	   -e DISPLAY=$DISPLAY \
	   --name warriorbot \
	   -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	   -v $(pwd)/ros2_ws:/WarriorBot_dev/ros2_ws \
	   --privileged \
	   warriorbot:latest bash
