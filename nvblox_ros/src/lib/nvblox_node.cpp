// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

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

using conversions::TsdfSliceType;

NvbloxNode::NvbloxNode(const rclcpp::NodeOptions& options,
                       const std::string& node_name)
    : Node(node_name, options), transformer_(this) {
  // Get parameters first (stuff below depends on parameters)
  getParameters();

  // Set the transformer settings.
  transformer_.set_global_frame(global_frame_);
  transformer_.set_pose_frame(pose_frame_);

  // Create callback groups, which allows processing to go in parallel with the
  // subscriptions.
  group_processing_ =
      create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  // Initialize the mapper (the interface to the underlying nvblox library)
  // Note: This needs to be called after getParameters()
  // The mapper includes:
  // - Map layers
  // - Integrators
  const std::string mapper_name = "mapper";
  declareMapperParameters(mapper_name, this);
  mapper_ = std::make_shared<Mapper>(voxel_size_, MemoryType::kDevice,
                                     static_projective_layer_type_);
  initializeMapper(mapper_name, mapper_.get(), this);

  mapper_->certified_mapping_enabled = use_certified_tsdf_;
  LOG(INFO) << "Certified mapping enabled: "
            << mapper_->certified_mapping_enabled;
  // TODO(rgg): set deallocation for fully deflated blocks

  // mark initial free area
  if (mark_free_sphere_radius_ > 0) {
    Vector3f mark_free_sphere_center(mark_free_sphere_center_x_,
                                     mark_free_sphere_center_y_,
                                     mark_free_sphere_center_z_);

    mapper_.get()->markUnobservedTsdfFreeInsideRadius(mark_free_sphere_center,
                                                      mark_free_sphere_radius_);
  }

  // Setup interactions with ROS
  subscribeToTopics();
  setupTimers();
  advertiseTopics();
  advertiseServices();

  // Start the message statistics
  depth_frame_statistics_.Start();
  rgb_frame_statistics_.Start();
  pointcloud_frame_statistics_.Start();

  RCLCPP_INFO_STREAM(get_logger(), "Started up nvblox node in frame "
                                       << global_frame_ << " and voxel size "
                                       << voxel_size_);

  // Set state.
  last_depth_update_time_ = rclcpp::Time(0ul, get_clock()->get_clock_type());
  last_color_update_time_ = rclcpp::Time(0ul, get_clock()->get_clock_type());
  last_lidar_update_time_ = rclcpp::Time(0ul, get_clock()->get_clock_type());
}

void NvbloxNode::getParameters() {
  RCLCPP_INFO_STREAM(get_logger(), "NvbloxNode::getParameters()");

  const bool is_occupancy =
      declare_parameter<bool>("use_static_occupancy_layer", false);
  if (is_occupancy) {
    static_projective_layer_type_ = ProjectiveLayerType::kOccupancy;
    RCLCPP_INFO_STREAM(
        get_logger(),
        "static_projective_layer_type: occupancy "
        "(Attention: ESDF and Mesh integration is not yet implemented "
        "for occupancy.)");
  } else {
    static_projective_layer_type_ = ProjectiveLayerType::kTsdf;
    RCLCPP_INFO_STREAM(
        get_logger(),
        "static_projective_layer_type: TSDF"
        " (for occupancy set the use_static_occupancy_layer parameter)");
  }

  // Declare & initialize the parameters.
  voxel_size_ = declare_parameter<float>("voxel_size", voxel_size_);
  global_frame_ = declare_parameter<std::string>("global_frame", global_frame_);
  pose_frame_ = declare_parameter<std::string>("pose_frame", pose_frame_);
  is_realsense_data_ =
      declare_parameter<bool>("is_realsense_data", is_realsense_data_);
  compute_mesh_ = declare_parameter<bool>("compute_mesh", compute_mesh_);
  compute_esdf_ = declare_parameter<bool>("compute_esdf", compute_esdf_);
  esdf_2d_ = declare_parameter<bool>("esdf_2d", esdf_2d_);
  esdf_distance_slice_ =
      declare_parameter<bool>("esdf_distance_slice", esdf_distance_slice_);
  use_color_ = declare_parameter<bool>("use_color", use_color_);
  use_depth_ = declare_parameter<bool>("use_depth", use_depth_);
  use_lidar_ = declare_parameter<bool>("use_lidar", use_lidar_);
  use_certified_tsdf_ =
      declare_parameter<bool>("use_certified_tsdf", use_certified_tsdf_);
  esdf_slice_height_ =
      declare_parameter<float>("esdf_slice_height", esdf_slice_height_);
  esdf_2d_min_height_ =
      declare_parameter<float>("esdf_2d_min_height", esdf_2d_min_height_);
  esdf_2d_max_height_ =
      declare_parameter<float>("esdf_2d_max_height", esdf_2d_max_height_);
  esdf_3d_origin_frame_id_ = declare_parameter<std::string>(
      "esdf_3d_origin_frame_id", esdf_3d_origin_frame_id_);
  esdf_3d_pub_range_x_ =
      declare_parameter<float>("esdf_3d_pub_range_x", esdf_3d_pub_range_x_);
  esdf_3d_pub_range_y_ =
      declare_parameter<float>("esdf_3d_pub_range_y", esdf_3d_pub_range_y_);
  esdf_3d_pub_range_z_ =
      declare_parameter<float>("esdf_3d_pub_range_z", esdf_3d_pub_range_z_);

  lidar_width_ = declare_parameter<int>("lidar_width", lidar_width_);
  lidar_height_ = declare_parameter<int>("lidar_height", lidar_height_);
  lidar_vertical_fov_rad_ = declare_parameter<float>("lidar_vertical_fov_rad",
                                                     lidar_vertical_fov_rad_);
  slice_visualization_attachment_frame_id_ =
      declare_parameter<std::string>("slice_visualization_attachment_frame_id",
                                     slice_visualization_attachment_frame_id_);
  slice_visualization_side_length_ = declare_parameter<float>(
      "slice_visualization_side_length", slice_visualization_side_length_);

  line_decomp_x_ = declare_parameter<float>("line_decomp_x", line_decomp_x_);
  line_decomp_y_ = declare_parameter<float>("line_decomp_y", line_decomp_y_);
  line_decomp_z_ = declare_parameter<float>("line_decomp_z", line_decomp_z_);

  // Update rates
  max_depth_update_hz_ =
      declare_parameter<float>("max_depth_update_hz", max_depth_update_hz_);
  max_color_update_hz_ =
      declare_parameter<float>("max_color_update_hz", max_color_update_hz_);
  max_lidar_update_hz_ =
      declare_parameter<float>("max_lidar_update_hz", max_lidar_update_hz_);
  mesh_update_rate_hz_ =
      declare_parameter<float>("mesh_update_rate_hz", mesh_update_rate_hz_);
  esdf_update_rate_hz_ =
      declare_parameter<float>("esdf_update_rate_hz", esdf_update_rate_hz_);
  esdf_3d_publish_rate_hz_ = declare_parameter<float>("esdf_3d_publish_rate_hz",
                                                      esdf_3d_publish_rate_hz_);
  occupancy_publication_rate_hz_ = declare_parameter<float>(
      "occupancy_publication_rate_hz", occupancy_publication_rate_hz_);
  max_poll_rate_hz_ =
      declare_parameter<float>("max_poll_rate_hz", max_poll_rate_hz_);

  maximum_sensor_message_queue_length_ =
      declare_parameter<int>("maximum_sensor_message_queue_length",
                             maximum_sensor_message_queue_length_);

  // Settings for QoSr
  depth_qos_str_ = declare_parameter<std::string>("depth_qos", depth_qos_str_);
  color_qos_str_ = declare_parameter<std::string>("color_qos", color_qos_str_);

  // Settings for map clearing
  map_clearing_radius_m_ =
      declare_parameter<float>("map_clearing_radius_m", map_clearing_radius_m_);
  map_clearing_frame_id_ = declare_parameter<std::string>(
      "map_clearing_frame_id", map_clearing_frame_id_);
  clear_outside_radius_rate_hz_ = declare_parameter<float>(
      "clear_outside_radius_rate_hz", clear_outside_radius_rate_hz_);

  // Settings for marking the initial sphere as free
  // will only clear any area if radius > 0
  mark_free_sphere_radius_ = declare_parameter<float>(
      "mark_free_sphere_radius_m", mark_free_sphere_radius_);
  mark_free_sphere_center_x_ = declare_parameter<float>(
      "mark_free_sphere_center_x", mark_free_sphere_center_x_);
  mark_free_sphere_center_y_ = declare_parameter<float>(
      "mark_free_sphere_center_y", mark_free_sphere_center_y_);
  mark_free_sphere_center_z_ = declare_parameter<float>(
      "mark_free_sphere_center_z", mark_free_sphere_center_z_);
}

void NvbloxNode::subscribeToTopics() {
  RCLCPP_INFO_STREAM(get_logger(), "NvbloxNode::subscribeToTopics()");

  constexpr int kQueueSize = 10;

  if (!use_depth_ && !use_lidar_) {
    RCLCPP_WARN(
        get_logger(),
        "Nvblox is running without depth or lidar input, the cost maps and"
        " reconstructions will not update");
  }

  if (use_depth_) {
    // Subscribe to synchronized depth + cam_info topics
    depth_sub_.subscribe(this, "depth/image", parseQosString(depth_qos_str_));
    depth_camera_info_sub_.subscribe(this, "depth/camera_info",
                                     parseQosString(depth_qos_str_));
    // TODO(rgg): subscribe to rototranslation error topic?
    timesync_depth_.reset(new message_filters::Synchronizer<time_policy_t>(
        time_policy_t(kQueueSize), depth_sub_, depth_camera_info_sub_));
    timesync_depth_->registerCallback(std::bind(&NvbloxNode::depthImageCallback,
                                                this, std::placeholders::_1,
                                                std::placeholders::_2));
  }
  if (use_color_) {
    // Subscribe to synchronized color + cam_info topics
    color_sub_.subscribe(this, "color/image", parseQosString(color_qos_str_));
    color_camera_info_sub_.subscribe(this, "color/camera_info",
                                     parseQosString(color_qos_str_));

    timesync_color_.reset(new message_filters::Synchronizer<time_policy_t>(
        time_policy_t(kQueueSize), color_sub_, color_camera_info_sub_));
    timesync_color_->registerCallback(std::bind(&NvbloxNode::colorImageCallback,
                                                this, std::placeholders::_1,
                                                std::placeholders::_2));
  }

  if (use_lidar_) {
    // Subscribe to pointclouds.
    pointcloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "pointcloud", kQueueSize,
        std::bind(&NvbloxNode::pointcloudCallback, this,
                  std::placeholders::_1));
  }

  // Subscribe to transforms.
  transform_sub_ = create_subscription<geometry_msgs::msg::TransformStamped>(
      "transform", kQueueSize,
      std::bind(&Transformer::transformCallback, &transformer_,
                std::placeholders::_1));
  pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "pose", 10,
      std::bind(&Transformer::poseCallback, &transformer_,
                std::placeholders::_1));
  if (use_certified_tsdf_) {
    pose_with_error_sub_ = create_subscription<
        certified_perception_msgs::msg::PoseWithErrorStamped>(
        "pose_with_error", 10,
        std::bind(&NvbloxNode::poseWithErrorCallback, this,
                  std::placeholders::_1));
  }
}

void NvbloxNode::advertiseTopics() {
  RCLCPP_INFO_STREAM(get_logger(), "NvbloxNode::advertiseTopics()");

  mesh_publisher_ = create_publisher<nvblox_msgs::msg::Mesh>("~/mesh", 1);
  esdf_pointcloud_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>("~/esdf_pointcloud", 1);
  certified_esdf_pointcloud_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>(
          "~/certified_esdf_pointcloud", 1);
  // Initialize all publishers for the certified tsdf
  certified_tsdf_cert_distance_pointcloud_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>(
          "~/certified_tsdf_cert_distance_pointcloud", 1);
  certified_tsdf_est_distance_pointcloud_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>(
          "~/certified_tsdf_est_distance_pointcloud", 1);
  certified_tsdf_correction_pointcloud_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>(
          "~/certified_tsdf_correction_pointcloud", 1);
  certified_tsdf_weight_pointcloud_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>(
          "~/certified_tsdf_weight_pointcloud", 1);
  esdfAABB_pointcloud_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>("~/esdfAABB_pointcloud",
                                                      1);
  map_slice_publisher_ =
      create_publisher<nvblox_msgs::msg::DistanceMapSlice>("~/map_slice", 1);
  certified_map_slice_publisher_ =
      create_publisher<nvblox_msgs::msg::DistanceMapSlice>("~/certified_map_slice", 1);
  mesh_marker_publisher_ =
      create_publisher<visualization_msgs::msg::MarkerArray>("~/mesh_marker",
                                                             1);
  mesh_pointcloud_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>("~/mesh_point_cloud", 1);
  slice_bounds_publisher_ = create_publisher<visualization_msgs::msg::Marker>(
      "~/map_slice_bounds", 1);
  occupancy_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>("~/occupancy", 1);
  sfc_publisher_ =
      create_publisher<decomp_ros_msgs::msg::PolyhedronStamped>("~/sfc", 1);
}

void NvbloxNode::advertiseServices() {
  RCLCPP_INFO_STREAM(get_logger(), "NvbloxNode::advertiseServices()");

  save_ply_service_ = create_service<nvblox_msgs::srv::FilePath>(
      "~/save_ply",
      std::bind(&NvbloxNode::savePly, this, std::placeholders::_1,
                std::placeholders::_2),
      rmw_qos_profile_services_default, group_processing_);
  save_ply_with_rotation_service_ = create_service<nvblox_msgs::srv::FilePath>(
      "~/save_ply_with_rotation",
      std::bind(&NvbloxNode::savePlyWithRotation, this, std::placeholders::_1,
                std::placeholders::_2),
      rmw_qos_profile_services_default, group_processing_);
  save_map_service_ = create_service<nvblox_msgs::srv::FilePath>(
      "~/save_map",
      std::bind(&NvbloxNode::saveMap, this, std::placeholders::_1,
                std::placeholders::_2),
      rmw_qos_profile_services_default, group_processing_);
  load_map_service_ = create_service<nvblox_msgs::srv::FilePath>(
      "~/load_map",
      std::bind(&NvbloxNode::loadMap, this, std::placeholders::_1,
                std::placeholders::_2),
      rmw_qos_profile_services_default, group_processing_);
}

void NvbloxNode::setupTimers() {
  RCLCPP_INFO_STREAM(get_logger(), "NvbloxNode::setupTimers()");
  if (use_depth_) {
    depth_processing_timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / max_poll_rate_hz_),
        std::bind(&NvbloxNode::processDepthQueue, this), group_processing_);
  }
  if (use_color_) {
    color_processing_timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / max_poll_rate_hz_),
        std::bind(&NvbloxNode::processColorQueue, this), group_processing_);
  }
  if (use_lidar_) {
    pointcloud_processing_timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / max_poll_rate_hz_),
        std::bind(&NvbloxNode::processPointcloudQueue, this),
        group_processing_);
  }
  if (use_certified_tsdf_) {
    pose_with_error_processing_timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / max_poll_rate_hz_),
        std::bind(&NvbloxNode::processPoseWithErrorQueue, this),
        group_processing_);
  }
  esdf_processing_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / esdf_update_rate_hz_),
      std::bind(&NvbloxNode::processEsdf, this), group_processing_);
  if (compute_esdf_ && !esdf_2d_) {
    esdf_3d_publish_timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / esdf_3d_publish_rate_hz_),
        std::bind(&NvbloxNode::publishEsdf3d, this), group_processing_);
  }
  mesh_processing_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / mesh_update_rate_hz_),
      std::bind(&NvbloxNode::processMesh, this), group_processing_);

  if (static_projective_layer_type_ == ProjectiveLayerType::kOccupancy) {
    occupancy_publishing_timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / occupancy_publication_rate_hz_),
        std::bind(&NvbloxNode::publishOccupancyPointcloud, this),
        group_processing_);
  }

  if (map_clearing_radius_m_ > 0.0f) {
    clear_outside_radius_timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / clear_outside_radius_rate_hz_),
        std::bind(&NvbloxNode::clearMapOutsideOfRadiusOfLastKnownPose, this),
        group_processing_);
  }
}

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

void NvbloxNode::poseWithErrorCallback(
    const certified_perception_msgs::msg::PoseWithErrorStamped::ConstSharedPtr
        pose_with_error) {
  printMessageArrivalStatistics(*pose_with_error, "Pose with Error Statistics",
                                &pose_with_error_frame_statistics_);
  pushMessageOntoQueue(pose_with_error, &pose_with_error_queue_,
                       &pose_with_error_queue_mutex_);
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

void NvbloxNode::processPoseWithErrorQueue() {
  using PoseWithErrorMsg =
      certified_perception_msgs::msg::PoseWithErrorStamped::ConstSharedPtr;
  // TODO(rgg): relevant in the context of poses?
  auto message_ready = [this](const PoseWithErrorMsg& msg) {
    return this->canTransform(
        msg->header);  // Replace with return true if needed.
  };
  processMessageQueue<PoseWithErrorMsg>(
      &pose_with_error_queue_, &pose_with_error_queue_mutex_, message_ready,
      std::bind(&NvbloxNode::processPoseWithError, this,
                std::placeholders::_1));
  // TODO(rgg): assess impact of removing rate limit, as if it occurs it will
  // compromise theoretical safety guarantee.
  limitQueueSizeByDeletingOldestMessages(
      maximum_sensor_message_queue_length_, "pose_with_error",
      &pose_with_error_queue_, &pose_with_error_queue_mutex_);
}

void NvbloxNode::processEsdf() {
  if (!compute_esdf_) {
    return;
  }
  const rclcpp::Time timestamp = get_clock()->now();
  timing::Timer ros_total_timer("ros/total");
  timing::Timer ros_esdf_timer("ros/esdf");

  timing::Timer esdf_integration_timer("ros/esdf/integrate");
  std::vector<Index3D> updated_blocks;
  if (esdf_2d_) {
    updated_blocks = mapper_->updateEsdfSlice(
        esdf_2d_min_height_, esdf_2d_max_height_, esdf_slice_height_);
  } else {
    updated_blocks = mapper_->updateEsdf();
  }
  esdf_integration_timer.Stop();

  if (updated_blocks.empty()) {
    return;
  }

  int cert_esdf_blocks = mapper_->certified_esdf_layer().numAllocatedBlocks();

  timing::Timer esdf_output_timer("ros/esdf/output");

  // If anyone wants a slice
  if (esdf_distance_slice_ &&
      (esdf_pointcloud_publisher_->get_subscription_count() > 0 
       || map_slice_publisher_->get_subscription_count() > 0 
       || certified_map_slice_publisher_->get_subscription_count() > 0
       || certified_esdf_pointcloud_publisher_->get_subscription_count() > 0
       || certified_tsdf_cert_distance_pointcloud_publisher_->get_subscription_count() > 0
       || certified_tsdf_est_distance_pointcloud_publisher_->get_subscription_count() > 0
       || certified_tsdf_correction_pointcloud_publisher_->get_subscription_count() > 0
       || certified_tsdf_weight_pointcloud_publisher_->get_subscription_count() > 0
       )
      ) {
    // Get the slice as an image
    timing::Timer esdf_slice_compute_timer("ros/esdf/output/compute");
    AxisAlignedBoundingBox aabb;
    Image<float> map_slice_image;
    esdf_slice_converter_.distanceMapSliceImageFromLayer(
        mapper_->esdf_layer(), esdf_slice_height_, &map_slice_image, &aabb);
    esdf_slice_compute_timer.Stop();

    // LOG(INFO) << "Creating certified distance map slice";
    Image<float> certified_map_slice_image;
    certified_esdf_slice_converter_.distanceMapSliceImageFromLayer(
        mapper_->certified_esdf_layer(), esdf_slice_height_, &certified_map_slice_image, &aabb);

    // Slice pointcloud for RVIZ
    if (esdf_pointcloud_publisher_->get_subscription_count() > 0) {
      timing::Timer esdf_output_pointcloud_timer("ros/esdf/output/pointcloud");
      sensor_msgs::msg::PointCloud2 pointcloud_msg;
      esdf_slice_converter_.sliceImageToPointcloud(
          map_slice_image, aabb, esdf_slice_height_,
          mapper_->esdf_layer().voxel_size(), &pointcloud_msg);
      pointcloud_msg.header.frame_id = global_frame_;
      pointcloud_msg.header.stamp = get_clock()->now();
      esdf_pointcloud_publisher_->publish(pointcloud_msg);
    }

    // Also publish the map slice (costmap for nav2).
    if (map_slice_publisher_->get_subscription_count() > 0) {
      timing::Timer esdf_output_human_slice_timer("ros/esdf/output/slice");
      nvblox_msgs::msg::DistanceMapSlice map_slice_msg;
      esdf_slice_converter_.distanceMapSliceImageToMsg(
          map_slice_image, aabb, esdf_slice_height_, mapper_->voxel_size_m(),
          &map_slice_msg);
      map_slice_msg.header.frame_id = global_frame_;
      map_slice_msg.header.stamp = get_clock()->now();
      map_slice_publisher_->publish(map_slice_msg);
    }
    
    // Also publish the certified map slice (costmap for nav2).
    if (certified_map_slice_publisher_->get_subscription_count() > 0) {
      timing::Timer esdf_output_human_slice_timer("ros/certified_esdf/output/slice");
      nvblox_msgs::msg::DistanceMapSlice certified_map_slice_msg;
      certified_esdf_slice_converter_.distanceMapSliceImageToMsg(
          certified_map_slice_image, aabb, esdf_slice_height_,
          mapper_->voxel_size_m(), &certified_map_slice_msg);
      certified_map_slice_msg.header.frame_id = global_frame_;
      certified_map_slice_msg.header.stamp = get_clock()->now();
      // LOG(INFO) << "Publishing certified distance map slice";
      certified_map_slice_publisher_->publish(certified_map_slice_msg);
    }

    // Slice certified ESDF pointcloud for RVIZ
    if (use_certified_tsdf_ && cert_esdf_blocks > 0 &&
        certified_esdf_pointcloud_publisher_->get_subscription_count() > 0) {
      // LOG(INFO) << "Publishing certified ESDF pointcloud";
      timing::Timer certified_esdf_output_pointcloud_timer(
          "ros/certified_esdf/output/pointcloud");
      sensor_msgs::msg::PointCloud2 pointcloud_msg;
      certified_esdf_slice_converter_.sliceImageToPointcloud(
          certified_map_slice_image, aabb, esdf_slice_height_,
          mapper_->certified_esdf_layer().voxel_size(), &pointcloud_msg);
      pointcloud_msg.header.frame_id = global_frame_;
      pointcloud_msg.header.stamp = get_clock()->now();
      certified_esdf_pointcloud_publisher_->publish(pointcloud_msg);
    }

    // Slice certified TSDF pointcloud for RVIZ
    // Make a list of publishers to loop over
    std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr>
        certified_tsdf_pointcloud_publishers{
            certified_tsdf_cert_distance_pointcloud_publisher_,
            certified_tsdf_est_distance_pointcloud_publisher_,
            certified_tsdf_correction_pointcloud_publisher_,
            certified_tsdf_weight_pointcloud_publisher_,
        };
    // Create corresponding list of slice types
    std::vector<TsdfSliceType> slice_types{
        TsdfSliceType::kCertifiedDistance, TsdfSliceType::kDistance,
        TsdfSliceType::kCorrection, TsdfSliceType::kWeight};
    // Iterate over the publishers and slice types
    for (size_t i = 0; i < certified_tsdf_pointcloud_publishers.size(); ++i) {
      // Get the publisher and slice type
      auto publisher = certified_tsdf_pointcloud_publishers[i];
      auto slice_type = slice_types[i]; 
      if (use_certified_tsdf_ &&
          publisher->get_subscription_count() > 0) {
        Image<float> cert_map_slice_image;
        certified_tsdf_slice_converter_.distanceMapSliceImageFromLayer(
            mapper_->certified_tsdf_layer(), esdf_slice_height_,
            &cert_map_slice_image, &aabb, slice_type);
        sensor_msgs::msg::PointCloud2 pointcloud_msg;
        certified_tsdf_slice_converter_.sliceImageToPointcloud(
            cert_map_slice_image, aabb, esdf_slice_height_,
            mapper_->certified_tsdf_layer().voxel_size(), &pointcloud_msg);
        pointcloud_msg.header.frame_id = global_frame_;
        pointcloud_msg.header.stamp = get_clock()->now();
        publisher->publish(pointcloud_msg);
      }
    }
  }

  // Also publish the slice bounds (showing esdf max/min 2d height)
  if (slice_bounds_publisher_->get_subscription_count() > 0) {
    // The frame to which the slice limits visualization is attached.
    // We get the transform from the plane-body (PB) frame, to the scene (S).
    Transform T_S_PB;
    if (transformer_.lookupTransformToGlobalFrame(
            slice_visualization_attachment_frame_id_, rclcpp::Time(0),
            &T_S_PB)) {
      // Get and publish the planes representing the slice bounds in z.
      const visualization_msgs::msg::Marker marker = sliceLimitsToMarker(
          T_S_PB, slice_visualization_side_length_, timestamp, global_frame_,
          esdf_2d_min_height_, esdf_2d_max_height_);
      slice_bounds_publisher_->publish(marker);
    } else {
      constexpr float kTimeBetweenDebugMessages = 1.0;
      RCLCPP_INFO_STREAM_THROTTLE(
          get_logger(), *get_clock(), kTimeBetweenDebugMessages,
          "Tried to publish slice bounds but couldn't look up frame: "
              << slice_visualization_attachment_frame_id_);
    }
  }
}

void NvbloxNode::publishEsdf3d() {
  if (!compute_esdf_ || esdf_2d_) {
    return;
  }
  const rclcpp::Time timestamp = get_clock()->now();
  timing::Timer ros_total_timer("ros/total");
  timing::Timer ros_esdf_timer("ros/esdf");
  timing::Timer esdf_output_timer("ros/esdf/output");

  // If anyone wants the AABB esdf, publish it here
  if ((esdfAABB_pointcloud_publisher_->get_subscription_count() > 0) ||
      (sfc_publisher_->get_subscription_count() > 0)) {
    timing::Timer esdf_3d_output_timer("ros/esdf/output/3d_pointcloud");
    // first look up the transform
    Transform T_L_EO;  // EO = esdf publisher origin frame id
    if (transformer_.lookupTransformToGlobalFrame(esdf_3d_origin_frame_id_,
                                                  rclcpp::Time(0), &T_L_EO)) {
      // got the transform successfully
      Vector3f esdf_3d_pub_origin = T_L_EO.translation();
      Vector3f esdf_3d_pub_range(esdf_3d_pub_range_x_, esdf_3d_pub_range_y_,
                                 esdf_3d_pub_range_z_);
      // create AABB
      AxisAlignedBoundingBox aabb_full(esdf_3d_pub_origin - esdf_3d_pub_range,
                                       esdf_3d_pub_origin + esdf_3d_pub_range);

      // chop off the ground
      auto aabb_min = aabb_full.min();
      // aabb_min(2) = 0.0f;
      AxisAlignedBoundingBox aabb(aabb_min, aabb_full.max());

      using PCLPoint = pcl::PointXYZI;
      using PCLPointCloud = pcl::PointCloud<PCLPoint>;
      PCLPointCloud pc;

      // populate the sensor message
      sensor_msgs::msg::PointCloud2 pointcloud_msg;
      layer_converter_.pointcloudMsgFromLayerInAABB(mapper_->esdf_layer(), aabb,
                                                    &pointcloud_msg);
      pointcloud_msg.header.frame_id = global_frame_;
      pointcloud_msg.header.stamp = get_clock()->now();

      if (esdfAABB_pointcloud_publisher_->get_subscription_count() > 0) {
        // only actually publish if someone wants the message
        esdfAABB_pointcloud_publisher_->publish(pointcloud_msg);
      }

      if (sfc_publisher_->get_subscription_count() > 0) {
        // only do the sfc decomposition if someone wants the message

        timing::Timer ros_sfc_timer("ros/sfc");
        {
          timing::Timer sfc_pcl_conversion_timer("ros/sfc/pclconv");

          // now compute the sfc decomposition
          pcl::fromROSMsg(pointcloud_msg, pc);
          RCLCPP_DEBUG(get_logger(), "pcl has %zu points", pc.size());
        }

        // convert to decompros type
        vec_Vec3f obs;
        {
          timing::Timer sfc_obs_create_timer("ros/sfc/obs_copy");
          for (PCLPointCloud::const_iterator it = pc.begin(); it != pc.end();
               ++it) {
            obs.push_back(Vec3f(it->x, it->y, it->z));
          }
        }

        // setup the decomposition
        auto sfc_origin = 0.5 * (aabb.min() + aabb.max());
        auto sfc_range = 0.5 * (aabb.max() - aabb.min());
        auto sfc_plus_x =
            T_L_EO * Vector3f(line_decomp_x_, line_decomp_y_, line_decomp_z_);

        sfc_plus_x(2) = sfc_origin(
            2);  // correct for the z offset caused by chopping off the ground

        // RCLCPP_INFO(get_logger(), "sfc_origin (%f, %f,%f), plus_x (%f, %f,
        // %f)",  sfc_origin(0), sfc_origin(1), sfc_origin(2), sfc_plus_x(0),
        // sfc_plus_x(1), sfc_plus_x(2));

        // SeedDecomp3D decomp(sfc_origin.cast<double>());
        LineSegment3D decomp(sfc_origin.cast<double>(),
                             sfc_plus_x.cast<double>());
        decomp.set_obs(obs);
        decomp.set_local_bbox(sfc_range.cast<double>());
        decomp.dilate(0.005f);

        Polyhedron3D poly;
        {
          timing::Timer sfc_get_poly_timer("ros/sfc/construct_poly");
          poly = decomp.get_polyhedron();
        }

        {
          timing::Timer sfc_publish_poly_timer("ros/sfc/publish_poly");

          // publish the polyhedron
          decomp_ros_msgs::msg::PolyhedronStamped poly_msg;
          poly_msg.header = pointcloud_msg.header;

          for (const auto& hp : poly.hyperplanes()) {
            geometry_msgs::msg::Point point;
            geometry_msgs::msg::Vector3 normal;
            point.x = hp.p_(0);
            point.y = hp.p_(1);
            point.z = hp.p_(2);
            normal.x = hp.n_(0);
            normal.y = hp.n_(1);
            normal.z = hp.n_(2);

            poly_msg.poly.ps.push_back(point);
            poly_msg.poly.ns.push_back(normal);
          }

          sfc_publisher_->publish(poly_msg);
        }
      }

    } else {
      constexpr float kTimeBetweenDebugMessages = 1.0;
      RCLCPP_INFO_STREAM_THROTTLE(
          get_logger(), *get_clock(), kTimeBetweenDebugMessages,
          "Tried to publish 3d esdf but couldn't look up frame: "
              << esdf_3d_origin_frame_id_);
    }
  }
}

void NvbloxNode::processMesh() {
  if (!compute_mesh_) {
    return;
  }
  const rclcpp::Time timestamp = get_clock()->now();
  timing::Timer ros_total_timer("ros/total");
  timing::Timer ros_mesh_timer("ros/mesh");

  timing::Timer mesh_integration_timer("ros/mesh/integrate_and_color");
  const std::vector<Index3D> mesh_updated_list = mapper_->updateMesh();
  mesh_integration_timer.Stop();

  // In the case that some mesh blocks have been re-added after deletion, remove
  // them from the deleted list.
  for (const Index3D& idx : mesh_updated_list) {
    mesh_blocks_deleted_.erase(idx);
  }
  // Make a list to be published to rviz of blocks to be removed from the viz
  const std::vector<Index3D> mesh_blocks_to_delete(mesh_blocks_deleted_.begin(),
                                                   mesh_blocks_deleted_.end());
  mesh_blocks_deleted_.clear();

  bool should_publish = !mesh_updated_list.empty();

  // Publish the mesh updates.
  timing::Timer mesh_output_timer("ros/mesh/output");
  size_t new_subscriber_count = mesh_publisher_->get_subscription_count();
  if (new_subscriber_count > 0) {
    nvblox_msgs::msg::Mesh mesh_msg;
    // In case we have new subscribers, publish the ENTIRE map once.
    if (new_subscriber_count > mesh_subscriber_count_) {
      RCLCPP_INFO(get_logger(), "Got a new subscriber, sending entire map.");
      conversions::meshMessageFromMeshLayer(mapper_->mesh_layer(), &mesh_msg);
      mesh_msg.clear = true;
      should_publish = true;
    } else {
      conversions::meshMessageFromMeshBlocks(mapper_->mesh_layer(),
                                             mesh_updated_list, &mesh_msg,
                                             mesh_blocks_to_delete);
    }
    mesh_msg.header.frame_id = global_frame_;
    mesh_msg.header.stamp = timestamp;
    if (should_publish) {
      mesh_publisher_->publish(mesh_msg);
    }
  }
  mesh_subscriber_count_ = new_subscriber_count;

  // optionally publish the markers.
  if (mesh_marker_publisher_->get_subscription_count() > 0) {
    visualization_msgs::msg::MarkerArray marker_msg;
    bool minimal_msg = false;
    conversions::markerMessageFromMeshLayer(
        mapper_->mesh_layer(), global_frame_, &marker_msg, minimal_msg);
    mesh_marker_publisher_->publish(marker_msg);
  }

  // optionally publish the mesh pointcloud
  if (mesh_pointcloud_publisher_->get_subscription_count() > 0) {
	  int downsample=5;
	  bool publish_single=false;
    timing::Timer mesh_pc_timer("ros/mesh/output/createPCmsg");
    sensor_msgs::msg::PointCloud2 pc_msg;
    conversions::pointcloudMessageFromMeshLayer(mapper_->mesh_layer(),
                                                global_frame_, &pc_msg, 
						downsample, 
						publish_single);

    mesh_pc_timer.Stop();
    timing::Timer mesh_pc_pub_timer("ros/mesh/output/pubPCmsg");
    mesh_pointcloud_publisher_->publish(pc_msg);
  }

  mesh_output_timer.Stop();
}

bool NvbloxNode::canTransform(const std_msgs::msg::Header& header) {
  Transform T_L_C;
  return transformer_.lookupTransformToGlobalFrame(header.frame_id,
                                                   header.stamp, &T_L_C);
}

bool NvbloxNode::isUpdateTooFrequent(const rclcpp::Time& current_stamp,
                                     const rclcpp::Time& last_update_stamp,
                                     float max_update_rate_hz) {
  if (max_update_rate_hz > 0.0f &&
      (current_stamp - last_update_stamp).seconds() <
          1.0f / max_update_rate_hz) {
    return true;
  }
  return false;
}

bool NvbloxNode::processPoseWithError(
    const certified_perception_msgs::msg::PoseWithErrorStamped::ConstSharedPtr&
        pose_with_error) {
  // Don't bother processing pose with error if certified mapping is not
  // enabled. There are no other consumers.
  timing::Timer certified_tsdf_integration_timer("ros/certified_tsdf/deflate");
  if (use_certified_tsdf_) {
    // Extract actual pose (not PoseWithError). This is T_G_P (global to
    // pose). When T_P_S (pose to sensor) is identity, and the layer frame is
    // the global frame, T_G_P is also T_L_C (layer frame to camera).
    geometry_msgs::msg::Pose pose = pose_with_error->pose;
    Transform T_L_C =
        transformer_.poseToEigen(pose);  // This method could be static
    // Extract error information.
    float eps_R = pose_with_error->rotation_error;
    float eps_t = pose_with_error->translation_error;
    // Deflate the mapper's certified TSDF with the new pose and error
    // information
    if (eps_R > 0 || eps_t > 0) {
      constexpr float kTimeBetweenDebugMessages = 1000.0;
      RCLCPP_INFO_STREAM_THROTTLE(
          get_logger(), *get_clock(), kTimeBetweenDebugMessages,
          "Deflating certified TSDF with eps_R: " << eps_R << " and eps_t: "
                                                  << eps_t << ".");
      mapper_->deflateCertifiedTsdf(T_L_C, eps_R, eps_t);
    }
  }
  return true;
}

bool NvbloxNode::processDepthImage(
    const std::pair<sensor_msgs::msg::Image::ConstSharedPtr,
                    sensor_msgs::msg::CameraInfo::ConstSharedPtr>&
        depth_camera_pair) {
  timing::Timer ros_depth_timer("ros/depth");
  timing::Timer transform_timer("ros/depth/transform");

  // Message parts
  const sensor_msgs::msg::Image::ConstSharedPtr& depth_img_ptr =
      depth_camera_pair.first;
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg =
      depth_camera_pair.second;

  // Check that we're not updating more quickly than we should.
  if (isUpdateTooFrequent(depth_img_ptr->header.stamp, last_depth_update_time_,
                          max_depth_update_hz_)) {
    return true;
  }
  last_depth_update_time_ = depth_img_ptr->header.stamp;

  // Get the TF for this image.
  Transform T_L_C;
  std::string target_frame = depth_img_ptr->header.frame_id;

  if (!transformer_.lookupTransformToGlobalFrame(
          target_frame, depth_img_ptr->header.stamp, &T_L_C)) {
    RCLCPP_WARN(get_logger(), "COULD NOT GET TRANSFORM!!");
    return false;
  }
  transform_timer.Stop();

  timing::Timer conversions_timer("ros/depth/conversions");
  // Convert camera info message to camera object.
  Camera camera = conversions::cameraFromMessage(*camera_info_msg);

  // Convert the depth image.
  if (!conversions::depthImageFromImageMessage(depth_img_ptr, &depth_image_)) {
    RCLCPP_ERROR(get_logger(), "Failed to transform depth image.");
    return false;
  }
  conversions_timer.Stop();

  // Integrate
  timing::Timer integration_timer("ros/depth/integrate");
  // This currently also updates certified TSDF if enabled.
  mapper_->integrateDepth(depth_image_, T_L_C, camera);

  integration_timer.Stop();
  return true;
}

bool NvbloxNode::processColorImage(
    const std::pair<sensor_msgs::msg::Image::ConstSharedPtr,
                    sensor_msgs::msg::CameraInfo::ConstSharedPtr>&
        color_camera_pair) {
  timing::Timer ros_color_timer("ros/color");
  timing::Timer transform_timer("ros/color/transform");

  const sensor_msgs::msg::Image::ConstSharedPtr& color_img_ptr =
      color_camera_pair.first;
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg =
      color_camera_pair.second;

  // Check that we're not updating more quickly than we should.
  if (isUpdateTooFrequent(color_img_ptr->header.stamp, last_color_update_time_,
                          max_color_update_hz_)) {
    return true;
  }
  last_color_update_time_ = color_img_ptr->header.stamp;

  // Get the TF for this image.
  const std::string target_frame = color_img_ptr->header.frame_id;
  Transform T_L_C;

  if (!transformer_.lookupTransformToGlobalFrame(
          target_frame, color_img_ptr->header.stamp, &T_L_C)) {
    return false;
  }

  transform_timer.Stop();

  timing::Timer color_convert_timer("ros/color/conversion");

  // Convert camera info message to camera object.
  Camera camera = conversions::cameraFromMessage(*camera_info_msg);

  // Convert the color image.
  if (!conversions::colorImageFromImageMessage(color_img_ptr, &color_image_)) {
    RCLCPP_ERROR(get_logger(), "Failed to transform color image.");
    return false;
  }
  color_convert_timer.Stop();

  // Integrate.
  timing::Timer color_integrate_timer("ros/color/integrate");
  mapper_->integrateColor(color_image_, T_L_C, camera);
  color_integrate_timer.Stop();
  return true;
}

bool NvbloxNode::processLidarPointcloud(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr& pointcloud_ptr) {
  timing::Timer ros_lidar_timer("ros/lidar");
  timing::Timer transform_timer("ros/lidar/transform");

  // Check that we're not updating more quickly than we should.
  if (isUpdateTooFrequent(pointcloud_ptr->header.stamp, last_lidar_update_time_,
                          max_lidar_update_hz_)) {
    return true;
  }
  last_lidar_update_time_ = pointcloud_ptr->header.stamp;

  // Get the TF for this image.
  const std::string target_frame = pointcloud_ptr->header.frame_id;
  Transform T_L_C;

  if (!transformer_.lookupTransformToGlobalFrame(
          target_frame, pointcloud_ptr->header.stamp, &T_L_C)) {
    return false;
  }

  transform_timer.Stop();

  // LiDAR intrinsics model
  Lidar lidar(lidar_width_, lidar_height_, lidar_vertical_fov_rad_);

  // We check that the pointcloud is consistent with this LiDAR model
  // NOTE(alexmillane): If the check fails we return true which indicates that
  // this pointcloud can be removed from the queue even though it wasn't
  // integrated (because the intrisics model is messed up).
  // NOTE(alexmillane): Note that internally we cache checks, so each LiDAR
  // intrisics model is only tested against a single pointcloud. This is because
  // the check is expensive to perform.
  if (!pointcloud_converter_.checkLidarPointcloud(pointcloud_ptr, lidar)) {
    RCLCPP_ERROR_ONCE(get_logger(),
                      "LiDAR intrinsics are inconsistent with the received "
                      "pointcloud. Failing integration.");
    return true;
  }

  timing::Timer lidar_conversion_timer("ros/lidar/conversion");
  pointcloud_converter_.depthImageFromPointcloudGPU(pointcloud_ptr, lidar,
                                                    &pointcloud_image_);
  lidar_conversion_timer.Stop();

  timing::Timer lidar_integration_timer("ros/lidar/integration");

  mapper_->integrateLidarDepth(pointcloud_image_, T_L_C, lidar);
  lidar_integration_timer.Stop();

  return true;
}

void NvbloxNode::publishOccupancyPointcloud() {
  timing::Timer ros_total_timer("ros/total");
  timing::Timer esdf_output_timer("ros/occupancy/output");

  if (occupancy_publisher_->get_subscription_count() > 0) {
    sensor_msgs::msg::PointCloud2 pointcloud_msg;
    layer_converter_.pointcloudMsgFromLayer(mapper_->occupancy_layer(),
                                            &pointcloud_msg);
    pointcloud_msg.header.frame_id = global_frame_;
    pointcloud_msg.header.stamp = get_clock()->now();
    occupancy_publisher_->publish(pointcloud_msg);
  }
}

void NvbloxNode::clearMapOutsideOfRadiusOfLastKnownPose() {
  if (map_clearing_radius_m_ > 0.0f) {
    timing::Timer("ros/clear_outside_radius");
    Transform T_L_MC;  // MC = map clearing frame
    if (transformer_.lookupTransformToGlobalFrame(map_clearing_frame_id_,
                                                  rclcpp::Time(0), &T_L_MC)) {
      const std::vector<Index3D> blocks_cleared = mapper_->clearOutsideRadius(
          T_L_MC.translation(), map_clearing_radius_m_);
      // We keep track of the deleted blocks for publishing later.
      mesh_blocks_deleted_.insert(blocks_cleared.begin(), blocks_cleared.end());
    } else {
      constexpr float kTimeBetweenDebugMessages = 1.0;
      RCLCPP_INFO_STREAM_THROTTLE(
          get_logger(), *get_clock(), kTimeBetweenDebugMessages,
          "Tried to clear map outside of radius but couldn't look up frame: "
              << map_clearing_frame_id_);
    }
  }
}

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

// Register the node as a component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvblox::NvbloxNode)
