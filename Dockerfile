FROM ros:melodic

RUN apt update && apt install -y python-pip ros-melodic-cv-bridge

WORKDIR /catkin_ws/src/

RUN . /opt/ros/melodic/setup.sh && catkin_init_workspace && cd .. && catkin_make clean && catkin_make -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release && catkin_make install

RUN echo "export ROS_MASTER_URI=http://core:11311">>~/startup.sh
RUN echo ". /opt/ros/melodic/setup.sh && . devel/setup.sh">>~/startup.sh
RUN echo "roslaunch realsense2_camera rs_camera.launch">>~/startup.sh

RUN chmod +x ~/startup.sh
CMD [ "~/startup.sh" ]