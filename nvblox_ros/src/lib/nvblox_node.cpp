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


  // Enable certified mapping
  if (use_certified_tsdf_) {
    mapper_->enableCertifiedMapping(true);
  }
  // TODO(rgg): set deallocation for fully deflated blocks

  // mark initial free area
  markInitialSphereFree();

  // Setup interactions with ROS
  subscribeToTopics();
  setupTimers();
  advertiseTopics();
  advertiseServices();

  // Start the message statistics
  depth_frame_statistics_.Start();
  rgb_frame_statistics_.Start();
  pointcloud_frame_statistics_.Start();
  pose_with_relative_cov_statistics_.Start();

  RCLCPP_INFO_STREAM(get_logger(), "Started up nvblox node in frame "
                                       << global_frame_ << " and voxel size "
                                       << voxel_size_);

  // Set state.
  last_depth_update_time_ = rclcpp::Time(0ul, get_clock()->get_clock_type());
  last_color_update_time_ = rclcpp::Time(0ul, get_clock()->get_clock_type());
  last_lidar_update_time_ = rclcpp::Time(0ul, get_clock()->get_clock_type());
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

    // Creating certified distance map slice
    timing::Timer certified_esdf_slice_compute_timer("ros/certified_esdf/output/compute");
    AxisAlignedBoundingBox certified_aabb;
    Image<float> certified_map_slice_image;
    certified_esdf_slice_converter_.distanceMapSliceImageFromLayer(
        mapper_->certified_esdf_layer(), esdf_slice_height_,
        &certified_map_slice_image, &certified_aabb);
    certified_esdf_slice_compute_timer.Stop();

    // Slice pointcloud for RVIZ
    if (esdf_pointcloud_publisher_->get_subscription_count() > 0) {
      DLOG(INFO) << "Publishing ESDF pointcloud";
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
      DLOG(INFO) << "Publishing ESDF map slice";
      timing::Timer esdf_output_human_slice_timer("ros/esdf/output/slice");
      nvblox_msgs::msg::DistanceMapSlice map_slice_msg;
      esdf_slice_converter_.distanceMapSliceImageToMsg(
          map_slice_image, aabb, esdf_slice_height_, mapper_->voxel_size_m(),
          &map_slice_msg);
      map_slice_msg.header.frame_id = global_frame_;
      map_slice_msg.header.stamp = get_clock()->now();
      map_slice_publisher_->publish(map_slice_msg);
    }
    

    // Slice certified ESDF pointcloud for RVIZ
    if (use_certified_tsdf_ && cert_esdf_blocks > 0 && certified_map_slice_image.dataConstPtr() != nullptr && 
        certified_esdf_pointcloud_publisher_->get_subscription_count() > 0) {
      DLOG(INFO) << "Publishing certified ESDF pointcloud";
      timing::Timer certified_esdf_output_pointcloud_timer(
          "ros/certified_esdf/output/pointcloud");
      sensor_msgs::msg::PointCloud2 pointcloud_msg;
      certified_esdf_slice_converter_.sliceImageToPointcloud(
          certified_map_slice_image, certified_aabb, esdf_slice_height_,
          mapper_->certified_esdf_layer().voxel_size(), &pointcloud_msg);
      pointcloud_msg.header.frame_id = global_frame_;
      pointcloud_msg.header.stamp = get_clock()->now();
      certified_esdf_pointcloud_publisher_->publish(pointcloud_msg);
    }
    
    // Also publish the certified map slice (costmap for nav2).
    if (certified_map_slice_publisher_->get_subscription_count() > 0 && certified_map_slice_image.dataConstPtr() != nullptr) {
      DLOG(INFO) << "Publishing certified ESDF map slice";
      timing::Timer esdf_output_human_slice_timer("ros/certified_esdf/output/slice");
      nvblox_msgs::msg::DistanceMapSlice certified_map_slice_msg;
      certified_esdf_slice_converter_.distanceMapSliceImageToMsg(
          certified_map_slice_image, certified_aabb, esdf_slice_height_,
          mapper_->voxel_size_m(), &certified_map_slice_msg);
      certified_map_slice_msg.header.frame_id = global_frame_;
      certified_map_slice_msg.header.stamp = get_clock()->now();
      certified_map_slice_publisher_->publish(certified_map_slice_msg);
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
            &cert_map_slice_image, &certified_aabb, slice_type);

            // print some deets on the sliceImage
            DLOG(INFO) << "**Slice type: " << static_cast<int>(slice_type) << "\n"
                      << "**size: " << cert_map_slice_image.width() << "x" << cert_map_slice_image.height() << "\n" 
                      << "**certified_aabb: " << certified_aabb.min().transpose() << " to " << certified_aabb.max().transpose()
                      << ((cert_map_slice_image.dataConstPtr() == nullptr)  ? " **no data**" : "there is data");

        if (cert_map_slice_image.dataConstPtr() == nullptr) {
          // the map slice is empty! 
          // dont publish anything
          continue;
        }

        sensor_msgs::msg::PointCloud2 pointcloud_msg;
        certified_tsdf_slice_converter_.sliceImageToPointcloud(
            cert_map_slice_image, certified_aabb, esdf_slice_height_,
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

void NvbloxNode::processCertifiedMesh() {
  if (!compute_mesh_) {
    return;
  }

  if (!use_certified_tsdf_) {
    return;
  }

  if (certified_mesh_publisher_->get_subscription_count() > 0) {
      const rclcpp::Time timestamp = get_clock()->now();
      timing::Timer ros_total_timer("ros/total");
      timing::Timer ros_mesh_timer("ros/certified_mesh");

      // create the new mesh
      timing::Timer mesh_integration_timer("ros/certified_mesh/integrate");
      mapper_->generateCertifiedMesh();
      const std::vector<Index3D> mesh_updated_list = 
            mapper_->certified_mesh_layer().getAllBlockIndices();
      mesh_integration_timer.Stop();

      // send the whole mesh
      timing::Timer mesh_output_timer("ros/certified_mesh/output");
      nvblox_msgs::msg::Mesh certified_mesh_msg;
      conversions::meshMessageFromMeshLayer(mapper_->certified_mesh_layer(), &certified_mesh_msg);
      certified_mesh_msg.clear = true;
      certified_mesh_msg.header.frame_id = global_frame_;
      certified_mesh_msg.header.stamp = timestamp;
      certified_mesh_publisher_->publish(certified_mesh_msg);
      mesh_output_timer.Stop();
  }

}

bool NvbloxNode::processPoseWithRelativeCov(
    const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr&
        pose_with_relative_cov) {

  if (use_certified_tsdf_) {
    timing::Timer certified_tsdf_integration_timer("ros/certified_tsdf/deflate");

    // Extract actual pose. This is T_G_P (global to
    // pose). When T_P_S (pose to sensor) is identity, and the layer frame is
    // the global frame, T_G_P is also T_L_C (layer frame to camera).
    geometry_msgs::msg::Pose pose = pose_with_relative_cov->pose.pose;
    Transform T_L_C =
        transformer_.poseToEigen(pose); 

    // Extract covariance information.
    TransformCovariance relative_cov = transformer_.covToEigen(pose_with_relative_cov->pose.covariance);

    // check the matrix is close to symmetric
    if (!relative_cov.isApprox(relative_cov.transpose(), 1e-3)) {
      RCLCPP_WARN(get_logger(), "relative covariance is not symmetric");
      return false; // should I deflate everything?
    }

    // grab the min and max eigenvalue 
    Eigen::SelfAdjointEigenSolver<TransformCovariance> eigensolver(relative_cov);
    float min_eval = eigensolver.eigenvalues().minCoeff();
    float max_eval = eigensolver.eigenvalues().maxCoeff();

    constexpr float kTimeBetweenDebugMessages = 1000.0;
    RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), kTimeBetweenDebugMessages,
          "Deflating certified TSDF with covariance in [%1.2g, %1.2g]", min_eval, max_eval);

    mapper_->deflateCertifiedTsdf(T_L_C, relative_cov, certified_n_std_);
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

}  // namespace nvblox

// Register the node as a component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvblox::NvbloxNode)
