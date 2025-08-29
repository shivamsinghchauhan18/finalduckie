cmake_minimum_required(VERSION 3.0.2)
project(blueduckie_lane_following)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  cv_bridge
  image_transport
  duckietown_msgs
)

find_package(OpenCV REQUIRED)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs geometry_msgs cv_bridge image_transport duckietown_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/advanced_lane_following_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
