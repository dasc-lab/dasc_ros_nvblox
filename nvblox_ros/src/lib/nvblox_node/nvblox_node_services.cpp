
#include "nvblox_ros/nvblox_node.hpp"

#include <nvblox/io/mesh_io.h>
#include <nvblox/io/pointcloud_io.h>
#include <nvblox/utils/timing.h>

#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <nvblox_ros_common/qos.hpp>

#include "nvblox_ros/conversions/certified_esdf_slice_conversions.hpp"
#include "nvblox_ros/conversions/certified_tsdf_slice_conversions.hpp"
#include "nvblox_ros/visualization.hpp"

namespace nvblox {

// Helper function for ends with. :)
bool ends_with(const std::string& value, const std::string& ending) {
  if (ending.size() > value.size()) {
    return false;
  }
  return std::equal(ending.crbegin(), ending.crend(), value.crbegin(),
                    [](const unsigned char a, const unsigned char b) {
                      return std::tolower(a) == std::tolower(b);
                    });
}

void NvbloxNode::savePly(
    const std::shared_ptr<nvblox_msgs::srv::FilePath::Request> request,
    std::shared_ptr<nvblox_msgs::srv::FilePath::Response> response) {
  // If we get a full path, then write to that path.
  bool success = false;
  if (ends_with(request->file_path, ".ply")) {
    success =
        io::outputMeshLayerToPly(mapper_->mesh_layer(), request->file_path);
  } else {
    // If we get a partial path then output a bunch of stuff to a folder.
    io::outputVoxelLayerToPly(mapper_->tsdf_layer(),
                              request->file_path + "/ros2_tsdf.ply");
    io::outputVoxelLayerToPly(mapper_->esdf_layer(),
                              request->file_path + "/ros2_esdf.ply");
    success = io::outputMeshLayerToPly(mapper_->mesh_layer(),
                                       request->file_path + "/ros2_mesh.ply");
  }
  if (success) {
    RCLCPP_INFO_STREAM(get_logger(),
                       "Output PLY file(s) to " << request->file_path);
    response->success = true;
  } else {
    RCLCPP_WARN_STREAM(get_logger(),
                       "Failed to write PLY file(s) to " << request->file_path);
    response->success = false;
  }
}

void NvbloxNode::savePlyWithRotation(
    const std::shared_ptr<nvblox_msgs::srv::FilePath::Request> request,
    std::shared_ptr<nvblox_msgs::srv::FilePath::Response> response) {
  // If we get a full path, then write to that path.
  if (ends_with(request->file_path, ".ply")) {
    response->success = false;
    return;
  }

  // If we get a partial path then output a bunch of stuff to a folder.
  // bool success_tsdf = io::outputVoxelLayerToPly(mapper_->tsdf_layer(),
  //                           request->file_path + "/ros2_tsdf.ply");
  bool success_esdf = io::outputVoxelLayerToPly(
      mapper_->esdf_layer(), request->file_path + "/ros2_esdf.ply");
  bool success_certified_esdf = io::outputVoxelLayerToPly(
      mapper_->certified_esdf_layer(),
      request->file_path + "/ros2_certified_esdf.ply");

  bool success_rotation =
      outputRototranslationToFile(request->file_path + "/rototranslation.txt");

  bool success = success_esdf && success_certified_esdf && success_rotation;

  if (success) {
    RCLCPP_INFO_STREAM(get_logger(),
                       "Output PLY file(s) to " << request->file_path);
    response->success = true;
  } else {
    RCLCPP_WARN_STREAM(
        get_logger(),
        "Failed to write PLY file(s) to "
            << request->file_path << ". Success_ESDF: " << success_esdf
            << " Success_Certified_ESDF: " << success_certified_esdf
            << " Success_Rototranslation: " << success_rotation);
    response->success = false;
  }
}

bool NvbloxNode::outputRototranslationToFile(const std::string& filename) {
  // grab the rotation
  Transform T_L_EO;  // EO = esdf publisher origin frame id
  // T_L_EO * p_B will return the point expressed in the B (body) frame as a
  // point in the map frame
  if (!transformer_.lookupTransformToGlobalFrame(esdf_3d_origin_frame_id_,
                                                 rclcpp::Time(0), &T_L_EO)) {
    // could not lookup transform, so return false
    RCLCPP_WARN(get_logger(), "Could not lookup rototranslation");
    return false;
  }

  // open the file
  std::ofstream file(filename);
  if (!file) {
    RCLCPP_WARN(get_logger(),
                "Could not open file for rototranslation writing");
    file.close();
    return false;
  }

  // write out the homogenous transform from map to bodyframe
  auto M = T_L_EO.matrix();  // get the matrix
  file << M;

  // close the file
  file.close();

  return true;
}

void NvbloxNode::saveMap(
    const std::shared_ptr<nvblox_msgs::srv::FilePath::Request> request,
    std::shared_ptr<nvblox_msgs::srv::FilePath::Response> response) {
  std::unique_lock<std::mutex> lock1(depth_queue_mutex_);
  std::unique_lock<std::mutex> lock2(color_queue_mutex_);

  std::string filename = request->file_path;
  if (!ends_with(request->file_path, ".nvblx")) {
    filename += ".nvblx";
  }

  response->success = mapper_->saveMap(filename);
  if (response->success) {
    RCLCPP_INFO_STREAM(get_logger(), "Output map to file to " << filename);
  } else {
    RCLCPP_WARN_STREAM(get_logger(), "Failed to write file to " << filename);
  }
}

void NvbloxNode::loadMap(
    const std::shared_ptr<nvblox_msgs::srv::FilePath::Request> request,
    std::shared_ptr<nvblox_msgs::srv::FilePath::Response> response) {
  std::unique_lock<std::mutex> lock1(depth_queue_mutex_);
  std::unique_lock<std::mutex> lock2(color_queue_mutex_);

  std::string filename = request->file_path;
  if (!ends_with(request->file_path, ".nvblx")) {
    filename += ".nvblx";
  }

  response->success = mapper_->loadMap(filename);
  if (response->success) {
    RCLCPP_INFO_STREAM(get_logger(), "Loaded map to file from " << filename);
  } else {
    RCLCPP_WARN_STREAM(get_logger(),
                       "Failed to load map file from " << filename);
  }
}

}  // namespace nvblox