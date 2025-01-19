cmake_minimum_required(VERSION 3.8)
project(distance_map_to_pointcloud)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nvblox_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(MPI REQUIRED)
if(MPI_FOUND)
  include_directories(${MPI_INCLUDE_PATH})
  link_directories(${MPI_LIBRARIES})
endif()


set(CMAKE_CXX_STANDARD 17)

add_executable(distance_map_to_pointcloud src/distance_map_to_pointcloud.cpp)
ament_target_dependencies(distance_map_to_pointcloud 
	rclcpp 
	sensor_msgs 
	nvblox_msgs
	pcl_conversions
	)

install(TARGETS distance_map_to_pointcloud
  DESTINATION lib/${PROJECT_NAME})

ament_package()

