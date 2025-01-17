#pragma once

// this file contains a set of utilities mostly related to eigen

#include <Eigen/Dense>
#include <random>

using Transform = Eigen::Isometry3d;
using CovMatrix = Eigen::Matrix<double, 6, 6>;
using AdjMatrix = Eigen::Matrix<double, 6, 6>;
using PoseWithCovMsg = geometry_msgs::msg::PoseWithCovarianceStamped;

template <typename MatrixType>
MatrixType make_symmetric(const MatrixType& M) {
  return 0.5 * (M + M.transpose());
}

template <typename ScalarT, int N>
Eigen::Matrix<ScalarT, N, N> matrix_sqrt(const Eigen::Matrix<ScalarT, N, N>& M,
                                         bool check_real = true) {
  // does not assume M is symmetric
  // assumes M has positive eigenvalues

  using CMatrixType = Eigen::Matrix<std::complex<ScalarT>, N, N>;
  using MatrixType = Eigen::Matrix<ScalarT, N, N>;
  using CVectorType = Eigen::Matrix<std::complex<ScalarT>, N, 1>;
  using VectorType = Eigen::Matrix<ScalarT, N, 1>;

  // compute the eigenvalues and eigenvectors
  Eigen::EigenSolver<MatrixType> solver(M);

  // extract result
  CVectorType complex_evals = solver.eigenvalues();
  CMatrixType complex_evecs = solver.eigenvectors();

  // if the eigvals check fails, return a zero matrix
  if (check_real && ((complex_evals.imag().norm() > 1e-6) ||
                     (complex_evecs.imag().norm() > 1e-3))) {
    std::cerr << "Ooooof. you gotta get real dude..." << std::endl;
    return MatrixType::Zero();
  }

  // extract the real part of the vector
  VectorType evals = complex_evals.real();
  MatrixType evecs = complex_evecs.real();

  // construct the matrix squareroot
  MatrixType sqrtM = evecs * evals.cwiseSqrt().asDiagonal() * evecs.inverse();

  return sqrtM;
}

template <typename ScalarT, int N, int M>
Eigen::Matrix<ScalarT, N, M> rand_matrix() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> dis(0.0, 1.0);

  // generate random matrix
  Eigen::Matrix<ScalarT, N, M> A;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      A(i, j) = dis(gen);
    }
  }

  return A;
}

template <typename ScalarT, int N>
Eigen::Matrix<ScalarT, N, N> rand_psd_matrix(ScalarT sigma_sq) {
  Eigen::Matrix<ScalarT, N, N> A = rand_matrix<ScalarT, N, N>();

  Eigen::Matrix<ScalarT, N, N> M = sigma_sq * A * A.transpose();

  return M;
}

CovMatrix rand_cov_matrix(double sigma_sq) {
  return rand_psd_matrix<double, 6>(sigma_sq);
}

Eigen::Matrix3d rand_rot_matrix(double max_angle = M_PI) {
  // Random number generators
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  std::uniform_real_distribution<> dis_angle(0.0, max_angle);

  // Generate a random vector and normalize it to create a random axis
  Eigen::Vector3d axis(dis(gen), dis(gen), dis(gen));
  axis.normalize();

  // Generate a random angle in the range [0, 2*pi)
  double angle = dis_angle(gen);

  // Create the rotation matrix using angle-axis representation
  Eigen::AngleAxisd angleAxis(angle, axis);
  Eigen::Matrix3d rotationMatrix = angleAxis.toRotationMatrix();

  return rotationMatrix;
}

Transform rand_transform(double max_angle, double sigma_sq_t) {
  Eigen::Matrix3d R = rand_rot_matrix(max_angle);
  Eigen::Vector3d t = sigma_sq_t * rand_matrix<double, 3, 1>();

  Transform T = Transform::Identity();
  T.linear() = R;
  T.translation() = t;
  return T;
}


