#include <rclcpp/rclcpp.hpp>
#include <nvblox_msgs/msg/distance_map_slice.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

class DistanceMapToPointCloudNode : public rclcpp::Node
{
public:
  DistanceMapToPointCloudNode() : Node("distance_map_to_pointcloud")
  {
    // Declare input topic
    declare_parameter<std::string>("input_topic", "/distance_map_slice");
    std::string input_topic = get_parameter("input_topic").as_string();


    std::string output_topic = get_parameter("input_topic").as_string() + "/pointcloud";

    // Initialize subscription and publisher
    distance_map_subscription_ = create_subscription<nvblox_msgs::msg::DistanceMapSlice>(
      input_topic, 10, std::bind(&DistanceMapToPointCloudNode::distanceMapCallback, this, std::placeholders::_1));

    point_cloud_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      output_topic, 10);

    RCLCPP_INFO(get_logger(), "Node started. Subscribing to: %s, publishing to: %s", input_topic.c_str(), output_topic.c_str());
  }

private:
  void distanceMapCallback(const nvblox_msgs::msg::DistanceMapSlice::SharedPtr msg)
  {
    // Create a PointCloud2 message
    auto point_cloud_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
    point_cloud_msg->header = msg->header;
    point_cloud_msg->height = 1; // Unordered point cloud
    point_cloud_msg->width = msg->data.size();
    point_cloud_msg->is_bigendian = false;
    point_cloud_msg->is_dense = false;

    // Define PointCloud2 fields (x, y, z, and distance)
    sensor_msgs::PointCloud2Modifier modifier(*point_cloud_msg);
    modifier.setPointCloud2FieldsByString(2, "xyz", "intensity");

    sensor_msgs::PointCloud2Iterator<float> iter_x(*point_cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*point_cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*point_cloud_msg, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_intensity(*point_cloud_msg, "intensity");

    // Fill in the PointCloud2 data
    for (size_t i = 0; i < msg->data.size(); ++i)
    {
      int x = i % msg->width;
      int y = i / msg->width;
      float distance = msg->data[i];

      // Skip unknown values
      if (distance == msg->unknown_value)
      {
        continue;
      }

      float wx = msg->origin.x + x * msg->resolution;
      float wy = msg->origin.y + y * msg->resolution;
      float wz = 0.0; // Assume slice is in the xy-plane

      *iter_x = wx;
      *iter_y = wy;
      *iter_z = wz;
      *iter_intensity = distance;

      ++iter_x;
      ++iter_y;
      ++iter_z;
      ++iter_intensity;
    }

    // Publish the PointCloud2 message
    point_cloud_publisher_->publish(*point_cloud_msg);
  }

  rclcpp::Subscription<nvblox_msgs::msg::DistanceMapSlice>::SharedPtr distance_map_subscription_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_publisher_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DistanceMapToPointCloudNode>());
  rclcpp::shutdown();
  return 0;
}

