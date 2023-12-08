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

#include <nvblox/gpu_hash/internal/cuda/gpu_indexing.cuh>

#include "nvblox_ros/conversions/certified_tsdf_slice_conversions.hpp"

namespace nvblox {
namespace conversions {

constexpr float kDistanceMapSliceUnknownValueCertTsdf = 1000.0f;

CertifiedTsdfSliceConverter::CertifiedTsdfSliceConverter() {
  cudaStreamCreate(&cuda_stream_);
}

__global__ void populateSliceFromLayerKernelCertTsdf(
    Index3DDeviceHashMapType<CertifiedTsdfBlock> block_hash,
    AxisAlignedBoundingBox aabb, float block_size, float* image, int rows,
    int cols, float z_slice_height, float resolution, float unobserved_value,
    TsdfSliceType slice_type) {
  const float voxel_size = block_size / CertifiedTsdfBlock::kVoxelsPerSide;
  const int pixel_col = blockIdx.x * blockDim.x + threadIdx.x;
  const int pixel_row = blockIdx.y * blockDim.y + threadIdx.y;

  if (pixel_col >= cols || pixel_row >= rows) {
    return;
  }

  // Figure out where this pixel should map to.
  Vector3f voxel_position(aabb.min().x() + resolution * pixel_col,
                          aabb.min().y() + resolution * pixel_row,
                          z_slice_height);

  Index3D block_index, voxel_index;

  getBlockAndVoxelIndexFromPositionInLayer(block_size, voxel_position,
                                           &block_index, &voxel_index);

  // Get the relevant block.
  CertifiedTsdfBlock* block_ptr = nullptr;
  auto it = block_hash.find(block_index);
  if (it != block_hash.end()) {
    block_ptr = it->second;
  } else {
    image::access(pixel_row, pixel_col, cols, image) = unobserved_value;
    return;
  }

  // Get the relevant pixel.
  const CertifiedTsdfVoxel* voxel =
      &block_ptr->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];
  // float distance = unobserved_value;
  // if (voxel->observed) {
  //   distance = voxel_size * std::sqrt(voxel->squared_distance_vox);
  //   if (voxel->is_inside) {
  //     distance = -distance;
  //   }
  // }

  // Based on the slice type, set the value to display
  float value{0.0f};
  if (slice_type == TsdfSliceType::kWeight) {
    value = voxel->weight;
  } else if (slice_type == TsdfSliceType::kCorrection) {
    value = voxel->correction;
  } else if (slice_type == TsdfSliceType::kDistance) {
    // Default to distance.
    // certified distance + correction = (observed) distance
    value = voxel->distance + voxel->correction;
  } else if (slice_type == TsdfSliceType::kCertifiedDistance) {
    // Default to certified distance as it is what's used by the ESDF
    value = voxel->distance;
  }
  image::access(pixel_row, pixel_col, cols, image) = value;
}

void CertifiedTsdfSliceConverter::populateSliceFromLayer(
    const CertifiedTsdfLayer& layer, const AxisAlignedBoundingBox& aabb,
    float z_slice_height, float resolution, float unobserved_value,
    Image<float>* image, TsdfSliceType slice_type) {
  CHECK(image->memory_type() == MemoryType::kDevice ||
        image->memory_type() == MemoryType::kUnified)
      << "Output needs to be accessible on device";
  // NOTE(alexmillane): At the moment we assume that the image is pre-allocated
  // to be the right size.
  CHECK_EQ(image->rows(),
           static_cast<int>(std::ceil(aabb.sizes().y() / resolution)));
  CHECK_EQ(image->cols(),
           static_cast<int>(std::ceil(aabb.sizes().x() / resolution)));
  if (image->numel() <= 0) {
    return;
  }
  const float voxel_size = layer.voxel_size();

  // Create a GPU hash of the ESDF.
  GPULayerView<CertifiedTsdfBlock> gpu_layer_view = layer.getGpuLayerView();

  // Pass in the GPU hash and AABB and let the kernel figure it out.
  constexpr int kThreadDim = 16;
  const int rounded_rows = static_cast<int>(
      std::ceil(image->rows() / static_cast<float>(kThreadDim)));
  const int rounded_cols = static_cast<int>(
      std::ceil(image->cols() / static_cast<float>(kThreadDim)));
  dim3 block_dim(rounded_cols, rounded_rows);
  dim3 thread_dim(kThreadDim, kThreadDim);
  populateSliceFromLayerKernelCertTsdf<<<block_dim, thread_dim, 0,
                                         cuda_stream_>>>(
      gpu_layer_view.getHash().impl_, aabb, layer.block_size(),
      image->dataPtr(), image->rows(), image->cols(), z_slice_height,
      resolution, unobserved_value, slice_type);
  checkCudaErrors(cudaStreamSynchronize(cuda_stream_));
  checkCudaErrors(cudaPeekAtLastError());
}

void CertifiedTsdfSliceConverter::distanceMapSliceImageFromLayer(
    const CertifiedTsdfLayer& layer, float z_slice_level,
    const AxisAlignedBoundingBox& aabb, Image<float>* map_slice_image_ptr,
    TsdfSliceType slice_type) {
  CHECK_NOTNULL(map_slice_image_ptr);

  const float voxel_size =
      layer.block_size() / VoxelBlock<CertifiedTsdfVoxel>::kVoxelsPerSide;
  Vector3f bounding_size = aabb.sizes();
  int width = static_cast<int>(std::ceil(bounding_size.x() / voxel_size));
  int height = static_cast<int>(std::ceil(bounding_size.y() / voxel_size));

  // Allocate a new image if required
  if (map_slice_image_ptr->rows() != height ||
      map_slice_image_ptr->cols() != width) {
    *map_slice_image_ptr = Image<float>(height, width, MemoryType::kDevice);
  }

  // Fill in the float image.
  populateSliceFromLayer(layer, aabb, z_slice_level, voxel_size,
                         kDistanceMapSliceUnknownValueCertTsdf,
                         map_slice_image_ptr, slice_type);
}

void CertifiedTsdfSliceConverter::distanceMapSliceImageFromLayer(
    const CertifiedTsdfLayer& layer, float z_slice_level,
    Image<float>* map_slice_image_ptr, AxisAlignedBoundingBox* aabb_ptr,
    TsdfSliceType slice_type) {
  CHECK_NOTNULL(map_slice_image_ptr);
  CHECK_NOTNULL(aabb_ptr);
  // Calls the AABB version of this function but first calculates the aabb

  // Get the AABB of the layer
  *aabb_ptr = getBoundingBoxOfLayerAtHeight(layer, z_slice_level);
  // LOG(INFO) << "AABB for slice: " << aabb_ptr->min().transpose() << " - "
  // << aabb_ptr->max().transpose();

  // Get the slice image
  distanceMapSliceImageFromLayer(layer, z_slice_level, *aabb_ptr,
                                 map_slice_image_ptr, slice_type);
}

void CertifiedTsdfSliceConverter::distanceMapSliceImageToMsg(
    const Image<float>& map_slice_image, const AxisAlignedBoundingBox& aabb,
    float z_slice_level, float voxel_size,
    nvblox_msgs::msg::DistanceMapSlice* map_slice) {
  CHECK_NOTNULL(map_slice);

  // Set up the message
  const int width = map_slice_image.cols();
  const int height = map_slice_image.rows();

  map_slice->origin.x = aabb.min().x();
  map_slice->origin.y = aabb.min().y();
  map_slice->origin.z = z_slice_level;

  map_slice->resolution = voxel_size;
  map_slice->width = width;
  map_slice->height = height;
  map_slice->unknown_value = kDistanceMapSliceUnknownValueCertTsdf;

  // Allocate the map directly, we will write directly to this output to prevent
  // copies.
  map_slice->data.resize(width * height, kDistanceMapSliceUnknownValueCertTsdf);

  // Copy into the message
  checkCudaErrors(
      cudaMemcpy(map_slice->data.data(), map_slice_image.dataConstPtr(),
                 map_slice_image.numel() * sizeof(float), cudaMemcpyDefault));
}

AxisAlignedBoundingBox
CertifiedTsdfSliceConverter::getBoundingBoxOfLayerAtHeight(
    const CertifiedTsdfLayer& layer, const float z_slice_level) {
  // Get the bounding box of the layer at this height
  // To do this, we get all the block indices, figure out which intersect
  // with our desired height, and select the min and max in the X and Y
  // direction.

  // Figure out the index of the desired height.
  Index3D desired_z_block_index;
  Index3D desired_z_voxel_index;
  getBlockAndVoxelIndexFromPositionInLayer(
      layer.block_size(), Vector3f(0.0f, 0.0f, z_slice_level),
      &desired_z_block_index, &desired_z_voxel_index);

  // Get a bounding box for the whole layer
  AxisAlignedBoundingBox aabb;
  aabb.setEmpty();
  for (const Index3D& block_index : layer.getAllBlockIndices()) {
    // Skip all other heights of block.
    if (block_index.z() != desired_z_block_index.z()) {
      continue;
    }
    // Extend the AABB by the dimensions of this block.
    aabb.extend(getAABBOfBlock(layer.block_size(), block_index));
  }
  return aabb;
}

void CertifiedTsdfSliceConverter::distanceMapSliceFromLayer(
    const CertifiedTsdfLayer& layer, float z_slice_level,
    nvblox_msgs::msg::DistanceMapSlice* map_slice) {
  CHECK_NOTNULL(map_slice);

  // Get the AABB of the layer
  const AxisAlignedBoundingBox aabb =
      getBoundingBoxOfLayerAtHeight(layer, z_slice_level);

  // Get the slice as an image
  Image<float> map_slice_image;
  distanceMapSliceImageFromLayer(layer, z_slice_level, aabb, &map_slice_image);

  // Convert slice image to a message
  distanceMapSliceImageToMsg(map_slice_image, aabb, z_slice_level,
                             layer.voxel_size(), map_slice);
}

void CertifiedTsdfSliceConverter::distanceMapSliceFromLayers(
    const CertifiedTsdfLayer& layer_1, const CertifiedTsdfLayer& layer_2,
    float z_slice_level, Image<float>* map_slice_image_ptr,
    AxisAlignedBoundingBox* aabb_ptr) {
  CHECK_NOTNULL(map_slice_image_ptr);

  // Combined (enclosing) AABB
  const AxisAlignedBoundingBox aabb_1 =
      getBoundingBoxOfLayerAtHeight(layer_1, z_slice_level);
  const AxisAlignedBoundingBox aabb_2 =
      getBoundingBoxOfLayerAtHeight(layer_2, z_slice_level);
  *aabb_ptr = aabb_1.merged(aabb_2);

  Image<float> map_slice_image_1;
  distanceMapSliceImageFromLayer(layer_1, z_slice_level, *aabb_ptr,
                                 &map_slice_image_1);
  distanceMapSliceImageFromLayer(layer_2, z_slice_level, *aabb_ptr,
                                 map_slice_image_ptr);

  // Get the minimal distance between the two slices
  image::elementWiseMinInPlaceGPU(map_slice_image_1, map_slice_image_ptr);
}

// Calling rules:
// - Should be called with 2D grid of thread-blocks where the total number of
// threads in each dimension exceeds the number of pixels in each image
// dimension ie:
// - blockIdx.x * blockDim.x + threadIdx.x > cols
// - blockIdx.y * blockDim.y + threadIdx.y > rows
// - We assume that theres enough space in the pointcloud to store a point per
// pixel if required.
__global__ void sliceImageToPointcloudKernelCertTsdf(
    const float* slice_image_ptr,       // NOLINT
    const int rows,                     // NOLINT
    const int cols,                     // NOLINT
    const AxisAlignedBoundingBox aabb,  // NOLINT
    const float z_slice_level,          // NOLINT
    const float voxel_size,             // NOLINT
    PclPointXYZI* pointcloud_ptr,       // NOLINT
    int* max_index) {
  // Get the pixel addressed by this thread.
  const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (col_idx >= cols || row_idx >= rows) {
    return;
  }

  // Access the slice
  const float pixel_value =
      image::access(row_idx, col_idx, cols, slice_image_ptr);

  // If map slice not observed don't add this point
  constexpr float kEps = 1e-2;
  if (fabsf(pixel_value - kDistanceMapSliceUnknownValueCertTsdf) < kEps) {
    return;
  }

  // 3D point associated with this pixel.
  PclPointXYZI point;
  point.x = aabb.min().x() + voxel_size * col_idx;
  point.y = aabb.min().y() + voxel_size * row_idx;
  point.z = z_slice_level;

  // Set the points intensity
  point.intensity = pixel_value;

  // Write the point to the ouput
  const int next_index = atomicAdd(max_index, 1);
  PclPointXYZI* point_out = &pointcloud_ptr[next_index];
  *point_out = point;
}

void CertifiedTsdfSliceConverter::sliceImageToPointcloud(
    const Image<float>& map_slice_image, const AxisAlignedBoundingBox& aabb,
    float z_slice_level, float voxel_size,
    sensor_msgs::msg::PointCloud2* pointcloud_msg) {
  CHECK_NOTNULL(pointcloud_msg);
  // LOG(INFO) << "Slice image to pointcloud";
  // LOG(INFO) << "Slice image size: " << map_slice_image.rows() << " x "
  //           << map_slice_image.cols();
  // LOG(INFO) << "Slice image numel: " << map_slice_image.numel();
  // LOG(INFO) << "z_slice_level: " << z_slice_level;
  // LOG(INFO) << "voxel_size: " << voxel_size;

  // Allocate max space we could take up
  pcl_pointcloud_device_.reserve(map_slice_image.numel());

  // Allocate output space for the number of valid points
  if (!max_index_device_) {
    max_index_device_ = make_unified<int>(MemoryType::kDevice);
  }
  max_index_device_.setZero();

  // Kernel
  // Call params
  // - 1 thread per pixel
  // - 8 x 8 threads per thread block
  // - N x M thread blocks get 1 thread per pixel
  constexpr dim3 kThreadsPerThreadBlock(8, 8, 1);
  const dim3 num_blocks(
      map_slice_image.cols() / kThreadsPerThreadBlock.x + 1,  // NOLINT
      map_slice_image.rows() / kThreadsPerThreadBlock.y + 1,  // NOLINT
      1);
  sliceImageToPointcloudKernelCertTsdf<<<num_blocks, kThreadsPerThreadBlock, 0,
                                         cuda_stream_>>>(
      map_slice_image.dataConstPtr(), map_slice_image.rows(),
      map_slice_image.cols(), aabb, z_slice_level, voxel_size,
      pcl_pointcloud_device_.data(), max_index_device_.get());
  checkCudaErrors(cudaStreamSynchronize(cuda_stream_));
  checkCudaErrors(cudaPeekAtLastError());

  // Retrieve how many points were actually used
  max_index_host_ = max_index_device_.clone(MemoryType::kHost);

  // Resize the pointcloud
  pcl_pointcloud_device_.resize(*max_index_host_);

  // Put the points in the message
  copyDevicePointcloudToMsg(pcl_pointcloud_device_, pointcloud_msg);
}

}  // namespace conversions
}  // namespace nvblox
