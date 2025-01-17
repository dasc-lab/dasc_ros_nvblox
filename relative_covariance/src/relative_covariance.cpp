#include <Eigen/Dense>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <random>
#include <rclcpp/rclcpp.hpp>

#include <relative_covariance/relative_covariance.hpp>
#include <relative_covariance/utils.hpp>

class RelativeCovarianceNode : public rclcpp::Node {
 public:
  // constructor
  RelativeCovarianceNode() : Node("relative_covariance") {
    // Initialize the publisher
    publisher_ =
        this->create_publisher<PoseWithCovMsg>("/relative_pose_with_cov", 10);

    // Initialize the subscriber
    subscription_ = this->create_subscription<PoseWithCovMsg>(
        "/visual_slam/tracking/vo_pose_covariance", 10,
        std::bind(&RelativeCovarianceNode::callback, this,
                  std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(),
                "Node initialized and subscribing to "
                "/visual_slam/tracking/vo_pose_covariance");
  }

 private:
  // extract the pose as an Eigen object
  Transform extract_pose(const PoseWithCovMsg::SharedPtr msg) {
    const auto& pose = msg->pose.pose;

    Transform eigen_pose =
        Eigen::Translation3d(pose.position.x, pose.position.y,
                             pose.position.z) *
        Eigen::Quaterniond(pose.orientation.w, pose.orientation.x,
                           pose.orientation.y, pose.orientation.z);

    return eigen_pose;
  }

  // extract the covariance as an eigen matrix
  CovMatrix extract_covariance(const PoseWithCovMsg::SharedPtr msg) {
    // grab the covariance part
    const auto& cov = msg->pose.covariance;

    // fill the eigen matrix
    CovMatrix eigen_cov = CovMatrix::Zero();

    eigen_cov.setZero();

    // fill each row
    for (size_t i = 0; i < 36; ++i) {
      size_t row = i / 6;
      size_t col = i % 6;
      eigen_cov(row, col) = cov[i];
    }

    // force symmetric
    eigen_cov = make_symmetric(eigen_cov);

    return eigen_cov;
  }

  bool fill_transform(PoseWithCovMsg& msg, const Transform& transform) {
    // extract the quaternion
    Eigen::Quaterniond q(transform.linear());
    Eigen::Vector3d t(transform.translation());

    geometry_msgs::msg::Pose pose_msg;

    pose_msg.position.x = t(0);
    pose_msg.position.y = t(1);
    pose_msg.position.z = t(2);

    pose_msg.orientation.x = q.x();
    pose_msg.orientation.y = q.y();
    pose_msg.orientation.z = q.z();
    pose_msg.orientation.w = q.w();

    msg.pose.pose = pose_msg;

    return true;
  }

  bool fill_covariance(PoseWithCovMsg& msg, const CovMatrix& cov_matrix) {
    int k = 0;
    for (int row = 0; row < 6; row++) {
      for (int col = 0; col < 6; col++) {
        msg.pose.covariance[k] = cov_matrix(row, col);
        k++;
      }
    }

    return true;
  }

  void callback(const PoseWithCovMsg::SharedPtr msg) {
    RCLCPP_INFO(get_logger(), "got message");

    // Extract the pose and covariance
    Transform pose = extract_pose(msg);
    CovMatrix cov = extract_covariance(msg);

    if (!has_initialized) {
      last_pose_ = pose;
      last_cov_ = cov;
      has_initialized = true;
      return;
    }

    // initialize vars
    Transform relative_pose = Transform::Identity();
    CovMatrix relative_cov = CovMatrix::Zero();

    // run the calculation
    bool success = get_relative_pose_and_cov(relative_pose, relative_cov,
                                             last_pose_, last_cov_, pose, cov,
                                             correlation_coefficient_);
    if (!success) {
      std::cerr << "something went wrong ..." << std::endl;
    }

    // print the min eigval
    auto evals = relative_cov.eigenvalues();
    if (evals.imag().norm() > 1e-6) {
	    RCLCPP_WARN(get_logger(), "relative covariance has complex eigenvalues: ");
	    std::cout << evals.transpose() << std::endl;
    }

    RCLCPP_INFO(get_logger(), "min eigval of relative_cov is %f", relative_cov.eigenvalues().real().minCoeff());

    // publish the relative transform object
    publish(msg, relative_pose, relative_cov);

    // copy over the last info
    last_pose_ = pose;
    last_cov_ = cov;

    return;
  }

  void publish(const PoseWithCovMsg::SharedPtr msg,
               const Transform& relative_pose, const CovMatrix& relative_cov) {
    // uses the same header as the msg
    // publishes a new pose with cov message
    PoseWithCovMsg new_msg;
    new_msg.header = msg->header;
    new_msg.header.frame_id = publish_frame;
    fill_transform(new_msg, relative_pose);
    fill_covariance(new_msg, relative_cov);

    publisher_->publish(new_msg);
    RCLCPP_INFO(get_logger(), "published new message");
  }

 public:
  bool test() {
    // create Sigma1 and the cov
    Transform pose_1 = Transform::Identity();
    CovMatrix cov_1 = rand_cov_matrix(1e-3);

    Transform pose_12 = rand_transform(1e-5, 1e-6);

    // create Sigma2 and the cov
    Transform pose_2 = pose_1 * pose_12;
    CovMatrix cov_2 = cov_1;

    //  print the two
    std::cout << pose_1.matrix() << std::endl << std::endl;
    std::cout << cov_1.eigenvalues().transpose() << std::endl << std::endl;

    std::cout << pose_2.matrix() << std::endl << std::endl;
    std::cout << cov_2.eigenvalues().transpose() << std::endl << std::endl;

    // now run the compute step
    Transform relative_pose;
    CovMatrix relative_cov;
    double rho = 0.99;
    get_relative_pose_and_cov(relative_pose, relative_cov,
                                             pose_1, cov_1, pose_2, cov_2, rho);

    // print results
    std::cout << "relative_transform: \n" << relative_pose.matrix() << "\n\n";
    std::cout << "expected_transform: \n"
              << (pose_1.inverse() * pose_2).matrix() << "\n\n";

    std::cout << "relative_cov: \n" << relative_cov << "\n\n";

    // try to fill cov and extract cov
    PoseWithCovMsg msg;

    fill_transform(msg, relative_pose);
    fill_covariance(msg, relative_cov);

    auto msg_ptr = std::make_shared<PoseWithCovMsg>(msg);

    Transform relative_tf_msg = extract_pose(msg_ptr);
    CovMatrix relative_cov_msg = extract_covariance(msg_ptr);

    std::cout << "fill and extract check transform: \n"
              << relative_pose.matrix() - relative_tf_msg.matrix() << "\n\n";
    std::cout << "fill and extract check covariance: \n"
              << relative_cov - relative_cov_msg << "\n\n";

    return true;
  }

 private:
  // PARAMETERS
  double correlation_coefficient_ = 0.99;
  std::string publish_frame = "body";  // TODO: make this a parameter

  // PRIVATE VARS:
  Transform last_pose_ = Transform::Identity();
  CovMatrix last_cov_ = CovMatrix::Zero();

  bool has_initialized = false;
  rclcpp::Subscription<PoseWithCovMsg>::SharedPtr subscription_;
  rclcpp::Publisher<PoseWithCovMsg>::SharedPtr publisher_;
};

int main(int argc, char* argv[]) {
  bool run_tests = true;

  rclcpp::init(argc, argv);
  auto node = std::make_shared<RelativeCovarianceNode>();

  if (run_tests) {
    node->test();
  } else {
    rclcpp::spin(node);
  }
  rclcpp::shutdown();
  return 0;
}
