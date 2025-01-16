cmake_minimum_required(VERSION 3.8)
project(distance_map_to_pointcloud)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nvblox_msgs REQUIRED)

set(CMAKE_CXX_STANDARD 17)

add_executable(distance_map_to_pointcloud src/distance_map_to_pointcloud.cpp)
ament_target_dependencies(distance_map_to_pointcloud 
	rclcpp 
	sensor_msgs 
	nvblox_msgs)

install(TARGETS distance_map_to_pointcloud
  DESTINATION lib/${PROJECT_NAME})

ament_package()

