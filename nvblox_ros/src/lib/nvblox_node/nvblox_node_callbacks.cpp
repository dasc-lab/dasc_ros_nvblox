
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

void NvbloxNode::depthImageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr& depth_img_ptr,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg) {
  printMessageArrivalStatistics(*depth_img_ptr, "Depth Statistics",
                                &depth_frame_statistics_);
  pushMessageOntoQueue({depth_img_ptr, camera_info_msg}, &depth_image_queue_,
                       &depth_queue_mutex_);
}

void NvbloxNode::colorImageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr& color_image_ptr,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg) {
  printMessageArrivalStatistics(*color_image_ptr, "Color Statistics",
                                &rgb_frame_statistics_);
  pushMessageOntoQueue({color_image_ptr, camera_info_msg}, &color_image_queue_,
                       &color_queue_mutex_);
}

void NvbloxNode::pointcloudCallback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr pointcloud) {
  printMessageArrivalStatistics(*pointcloud, "Pointcloud Statistics",
                                &pointcloud_frame_statistics_);
  pushMessageOntoQueue(pointcloud, &pointcloud_queue_,
                       &pointcloud_queue_mutex_);
}

// void NvbloxNode::poseWithErrorCallback(
//     const certified_perception_msgs::msg::PoseWithErrorStamped::ConstSharedPtr
//         pose_with_error) {
//   printMessageArrivalStatistics(*pose_with_error, "Pose with Error Statistics",
//                                 &pose_with_error_frame_statistics_);
//   pushMessageOntoQueue(pose_with_error, &pose_with_error_queue_,
//                        &pose_with_error_queue_mutex_);
// }

void NvbloxNode::poseWithRelativeCovCallback(
    const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr
        pose_with_relative_cov) {
  printMessageArrivalStatistics(*pose_with_relative_cov, "Pose with Relative Covariance Statistics",
                                &pose_with_relative_cov_statistics_);

  pushMessageOntoQueue(pose_with_relative_cov, &pose_with_relative_cov_queue_,
                       &pose_with_relative_cov_queue_mutex_);
}

void NvbloxNode::processDepthQueue() {
  using ImageInfoMsgPair =
      std::pair<sensor_msgs::msg::Image::ConstSharedPtr,
                sensor_msgs::msg::CameraInfo::ConstSharedPtr>;
  auto message_ready = [this](const ImageInfoMsgPair& msg) {
    return this->canTransform(msg.first->header);
  };

  processMessageQueue<ImageInfoMsgPair>(
      &depth_image_queue_,  // NOLINT
      &depth_queue_mutex_,  // NOLINT
      message_ready,        // NOLINT
      std::bind(&NvbloxNode::processDepthImage, this, std::placeholders::_1));

  limitQueueSizeByDeletingOldestMessages(maximum_sensor_message_queue_length_,
                                         "depth", &depth_image_queue_,
                                         &depth_queue_mutex_);
}

void NvbloxNode::processColorQueue() {
  using ImageInfoMsgPair =
      std::pair<sensor_msgs::msg::Image::ConstSharedPtr,
                sensor_msgs::msg::CameraInfo::ConstSharedPtr>;
  auto message_ready = [this](const ImageInfoMsgPair& msg) {
    return this->canTransform(msg.first->header);
  };

  processMessageQueue<ImageInfoMsgPair>(
      &color_image_queue_,  // NOLINT
      &color_queue_mutex_,  // NOLINT
      message_ready,        // NOLINT
      std::bind(&NvbloxNode::processColorImage, this, std::placeholders::_1));

  limitQueueSizeByDeletingOldestMessages(maximum_sensor_message_queue_length_,
                                         "color", &color_image_queue_,
                                         &color_queue_mutex_);
}

void NvbloxNode::processPointcloudQueue() {
  using PointcloudMsg = sensor_msgs::msg::PointCloud2::ConstSharedPtr;
  auto message_ready = [this](const PointcloudMsg& msg) {
    return this->canTransform(msg->header);
  };
  processMessageQueue<PointcloudMsg>(
      &pointcloud_queue_,        // NOLINT
      &pointcloud_queue_mutex_,  // NOLINT
      message_ready,             // NOLINT
      std::bind(&NvbloxNode::processLidarPointcloud, this,
                std::placeholders::_1));

  limitQueueSizeByDeletingOldestMessages(maximum_sensor_message_queue_length_,
                                         "pointcloud", &pointcloud_queue_,
                                         &pointcloud_queue_mutex_);
}


void NvbloxNode::processPoseWithRelativeCovQueue() {
  using PoseWithRelativeCovMsg =
      geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr;

  auto message_ready = [this](const PoseWithRelativeCovMsg& msg) {
    // return true;
    // dont bother checking that we can transform
    std::cout << "in processPoseWithRelativeCovQueue, checking if canTransform "  << msg->header.frame_id << std::endl;
    return this->canTransform(
        msg->header);
  };

  processMessageQueue<PoseWithRelativeCovMsg>(
      &pose_with_relative_cov_queue_, &pose_with_relative_cov_queue_mutex_, message_ready,
      std::bind(&NvbloxNode::processPoseWithRelativeCov, this,
                std::placeholders::_1));

  // TODO(rgg): assess impact of removing rate limit, as if it occurs it will
  // compromise theoretical safety guarantee.
  limitQueueSizeByDeletingOldestMessages(
      maximum_sensor_message_queue_length_, "pose_with_relative_cov",
      &pose_with_relative_cov_queue_, &pose_with_relative_cov_queue_mutex_);
}

}  // namespace nvblox