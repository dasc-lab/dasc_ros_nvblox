#pragma once

#include "utils.hpp"

AdjMatrix get_adjoint(const Transform& transform) {
  AdjMatrix A = AdjMatrix::Identity();

  Eigen::Matrix3d R = transform.linear();
  Eigen::Vector3d t = transform.translation();

  Eigen::Matrix3d t_hat;
  t_hat << 0, -t(2), t(1), t(2), 0, -t(0), -t(1), t(0), 0;

  A.block<3, 3>(0, 0) = R;
  A.block<3, 3>(3, 3) = R;
  A.block<3, 3>(0, 3) = t_hat * R;

  return A;
}

bool get_relative_pose_and_cov(Transform& relative_pose,
                               CovMatrix& relative_cov,
                               const Transform& last_pose,
                               const CovMatrix& last_cov, const Transform& pose,
                               const CovMatrix& cov,
                               double correlation_coefficient) {
  // compute the relative transform
  relative_pose = last_pose.inverse() * pose;

  // compute the cross term
  Eigen::Matrix<double, 6, 6> Sigma_cross_squared = last_cov * cov.transpose();
  Eigen::Matrix<double, 6, 6> Sigma_cross =
      correlation_coefficient * matrix_sqrt(Sigma_cross_squared);

  if (false) {
    // construct the full sigma matrix just to check things
    Eigen::Matrix<double, 12, 12> Sigma_full;
    Sigma_full.setZero();
    Sigma_full.block<6, 6>(0, 0) = last_cov;
    Sigma_full.block<6, 6>(6, 6) = cov;
    Sigma_full.block<6, 6>(0, 6) = Sigma_cross;
    Sigma_full.block<6, 6>(6, 0) = Sigma_cross.transpose();

    // std::cout << "Sigma full: \n" << Sigma_full << std::endl << std::endl;

    // std::cout << "is Sigma full symm?" << Sigma_full - Sigma_full.transpose()
    // << "\n\n";

    // check the eigenvalues of Sigma_full
    // std::cout << "evals(sigma_full): " << Sigma_full.eigenvalues().transpose()
    //           << "\n\n";
  }

  // get adjoint
  AdjMatrix A = get_adjoint(relative_pose).inverse();
  // std::cout << "A: \n" << A << "\n\n";

  // compute the incremental covariance
  CovMatrix Sigma = A * last_cov * A.transpose() + cov.matrix() -
                    A * Sigma_cross - (A * Sigma_cross).transpose();

  // make sure its a symmetric matrix
  relative_cov = make_symmetric(Sigma);

  // std::cout << "Sigma_relative: \n";
  // std::cout << relative_cov << std::endl;
  // std::cout << "relative_cov evals: \n";
  // std::cout << relative_cov.eigenvalues().transpose() << std::endl;

  return true;
}
