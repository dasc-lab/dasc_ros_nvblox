#include <rclcpp/rclcpp.hpp>
#include <nvblox_msgs/msg/distance_map_slice.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>

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

    publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      output_topic, 10);

    RCLCPP_INFO(get_logger(), "Node started. Subscribing to: %s, publishing to: %s", input_topic.c_str(), output_topic.c_str());
  }

private:
  void distanceMapCallback(const nvblox_msgs::msg::DistanceMapSlice::SharedPtr msg)
  {

	  // create a pcl pointcloud
	  pcl::PointCloud<pcl::PointXYZI> cloud;

	  std::size_t N = msg-> data.size();

	  RCLCPP_INFO(get_logger(), "Got %zu points", N);

	  for (std::size_t i=0; i< N; ++i)
	  {
		  // create the point
		  pcl::PointXYZI pt;
		  float x = i % msg->width;
		  float y = i % msg->width;
		  float d = msg->data[i];

		  // skip if unknown
		  if (d == msg -> unknown_value)
		  {
			  continue;
		  }
       pt.x = msg->origin.x + x * msg->resolution;
       pt.y = msg->origin.y + y * msg->resolution;
       pt.z = msg->origin.z; // Assume slice is in the xy-plane

       pt.intensity = d;

       cloud.push_back(pt);

	  }

	  // make a sensor message
	  auto pc_msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
	  pcl::toROSMsg(cloud, *pc_msg);
	  pc_msg->header= msg->header;

	  // publish it 
	  publisher_->publish(*pc_msg);


  }

  rclcpp::Subscription<nvblox_msgs::msg::DistanceMapSlice>::SharedPtr distance_map_subscription_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DistanceMapToPointCloudNode>());
  rclcpp::shutdown();
  return 0;
}

