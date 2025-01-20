
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
  certified_n_std_ = declare_parameter<float>("certified_n_std", certified_n_std_);
  deallocate_fully_deflated_blocks_ = declare_parameter<bool>(
      "deallocate_fully_deflated_blocks", deallocate_fully_deflated_blocks_);

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

  // subscribe to pose with relative covariance transform
  if (use_certified_tsdf_) {

    pose_with_relative_cov_sub_ = create_subscription<
        geometry_msgs::msg::PoseWithCovarianceStamped>(
        "pose_with_relative_covariance", kQueueSize,
        std::bind(&NvbloxNode::poseWithRelativeCovCallback, this, 
                  std::placeholders::_1));
  }
}

void NvbloxNode::advertiseTopics() {
  RCLCPP_INFO_STREAM(get_logger(), "NvbloxNode::advertiseTopics()");


  // Mesh publishers
  mesh_publisher_ = create_publisher<nvblox_msgs::msg::Mesh>("~/mesh", 1);
  certified_mesh_publisher_ = create_publisher<nvblox_msgs::msg::Mesh>("~/certified_mesh", 1);
  
  mesh_marker_publisher_ =
      create_publisher<visualization_msgs::msg::MarkerArray>("~/mesh_marker",
                                                             1);
  mesh_pointcloud_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>("~/mesh_point_cloud", 1);

  // esdf slice pointcloud publisher
  esdf_pointcloud_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>("~/esdf_pointcloud", 1);
  certified_esdf_pointcloud_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>(
          "~/certified_esdf_pointcloud", 1);

  // TDSF slice pointcloud publishers
  certified_tsdf_cert_distance_pointcloud_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>(
          "~/certified_tsdf_pointcloud/cert_distance", 1);
  certified_tsdf_est_distance_pointcloud_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>(
          "~/certified_tsdf_pointcloud/est_distance", 1);
  certified_tsdf_correction_pointcloud_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>(
          "~/certified_tsdf_pointcloud/correction", 1);
  certified_tsdf_weight_pointcloud_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>(
          "~/certified_tsdf_pointcloud/weight", 1);

  // 3D esdf pointcloud publisher
  esdfAABB_pointcloud_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>("~/esdfAABB_pointcloud",
                                                      1);
  
  // distance map slice publishers
  map_slice_publisher_ =
      create_publisher<nvblox_msgs::msg::DistanceMapSlice>("~/map_slice", 1);
  certified_map_slice_publisher_ =
      create_publisher<nvblox_msgs::msg::DistanceMapSlice>("~/certified_map_slice", 1);
  slice_bounds_publisher_ = create_publisher<visualization_msgs::msg::Marker>(
      "~/map_slice_bounds", 1);


  // some markers useful for nvblox viz without custom messages
  occupancy_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>("~/occupancy", 1);

  // publisher for the safe flight corridor extracted from a local esdf
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
    // // TODO(dev): replace with the relative pose covariance
    // pose_with_error_processing_timer_ = create_wall_timer(
    //     std::chrono::duration<double>(1.0 / max_poll_rate_hz_),
    //     std::bind(&NvbloxNode::processPoseWithErrorQueue, this),
    //     group_processing_);
    pose_with_relative_cov_processing_timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / max_poll_rate_hz_),
        std::bind(&NvbloxNode::processPoseWithRelativeCovQueue, this),
        group_processing_
    );
  }

  // run timers for the esdf processing
  esdf_processing_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / esdf_update_rate_hz_),
      std::bind(&NvbloxNode::processEsdf, this), group_processing_);


  // timers to publish the processed 3D Esdf
  if (compute_esdf_ && !esdf_2d_) {
    esdf_3d_publish_timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / esdf_3d_publish_rate_hz_),
        std::bind(&NvbloxNode::publishEsdf3d, this), group_processing_);
  }

  // timers for mesh construction
  mesh_processing_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / mesh_update_rate_hz_),
      std::bind(&NvbloxNode::processMesh, this), group_processing_);

  if (use_certified_tsdf_) {
    certified_mesh_processing_timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / mesh_update_rate_hz_),
        std::bind(&NvbloxNode::processCertifiedMesh, this), group_processing_);
  }

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

}  // namespace nvblox