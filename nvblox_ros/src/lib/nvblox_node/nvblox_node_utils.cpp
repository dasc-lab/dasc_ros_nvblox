
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

void NvbloxNode::markInitialSphereFree() {
  if (mark_free_sphere_radius_ > 0) {
    Vector3f mark_free_sphere_center(mark_free_sphere_center_x_,
                                     mark_free_sphere_center_y_,
                                     mark_free_sphere_center_z_);

    mapper_.get()->markUnobservedTsdfFreeInsideRadius(mark_free_sphere_center,
                                                      mark_free_sphere_radius_);
  }
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


}  // namespace nvblox