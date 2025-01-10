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

#include "nvblox_ros/conversions/mesh_conversions.hpp"

namespace nvblox
{
namespace conversions
{

geometry_msgs::msg::Point32 point32MessageFromVector(
  const Eigen::Vector3f & vector)
{
  geometry_msgs::msg::Point32 point;
  point.x = vector.x();
  point.y = vector.y();
  point.z = vector.z();
  return point;
}

geometry_msgs::msg::Point pointMessageFromVector(
  const Eigen::Vector3f & vector)
{
  geometry_msgs::msg::Point point;
  point.x = vector.x();
  point.y = vector.y();
  point.z = vector.z();
  return point;
}

std_msgs::msg::ColorRGBA colorMessageFromColor(const Color & color)
{
  std_msgs::msg::ColorRGBA color_msg;
  color_msg.r = static_cast<float>(color.r) / 255.0f;
  color_msg.g = static_cast<float>(color.g) / 255.0f;
  color_msg.b = static_cast<float>(color.b) / 255.0f;
  color_msg.a = 1.0f;
  return color_msg;
}

nvblox_msgs::msg::Index3D index3DMessageFromIndex3D(const Index3D & index)
{
  nvblox_msgs::msg::Index3D index_msg;
  index_msg.x = index.x();
  index_msg.y = index.y();
  index_msg.z = index.z();
  return index_msg;
}

void meshMessageFromMeshLayer(
  const BlockLayer<MeshBlock> & mesh_layer,
  nvblox_msgs::msg::Mesh * mesh_msg)
{
  std::vector<Index3D> block_indices = mesh_layer.getAllBlockIndices();
  meshMessageFromMeshBlocks(mesh_layer, block_indices, mesh_msg);
}

void meshBlockMessageFromMeshBlock(
  const MeshBlock & mesh_block, nvblox_msgs::msg::MeshBlock * mesh_block_msg)
{
  CHECK_NOTNULL(mesh_block_msg);

  size_t num_vertices = mesh_block.vertices.size();

  mesh_block_msg->vertices.resize(num_vertices);
  mesh_block_msg->normals.resize(num_vertices);
  mesh_block_msg->colors.resize(mesh_block.colors.size());
  mesh_block_msg->triangles.resize(mesh_block.triangles.size());

  std::vector<Vector3f> vertices = mesh_block.getVertexVectorOnCPU();
  std::vector<Vector3f> normals = mesh_block.getNormalVectorOnCPU();
  std::vector<Color> colors = mesh_block.getColorVectorOnCPU();

  // Copy over vertices and normals.
  for (size_t i = 0; i < num_vertices; i++) {
    mesh_block_msg->vertices[i] = point32MessageFromVector(vertices[i]);
    mesh_block_msg->normals[i] = point32MessageFromVector(normals[i]);
  }

  // Copy over colors if available.
  for (size_t i = 0; i < mesh_block.colors.size(); i++) {
    mesh_block_msg->colors[i] = colorMessageFromColor(colors[i]);
  }

  // Copying over triangles is thankfully easy.
  mesh_block_msg->triangles = mesh_block.getTriangleVectorOnCPU();
}

void meshMessageFromMeshBlocks(
  const BlockLayer<MeshBlock> & mesh_layer,
  const std::vector<Index3D> & block_indices, nvblox_msgs::msg::Mesh * mesh_msg,
  const std::vector<Index3D> & block_indices_to_delete)
{
  // Go through all the blocks, converting each individual one.
  mesh_msg->block_size = mesh_layer.block_size();
  mesh_msg->block_indices.resize(block_indices.size());
  mesh_msg->blocks.resize(block_indices.size());

  for (size_t i = 0; i < block_indices.size(); i++) {
    // Get the block origin.
    mesh_msg->block_indices[i] = index3DMessageFromIndex3D(block_indices[i]);

    MeshBlock::ConstPtr mesh_block =
      mesh_layer.getBlockAtIndex(block_indices[i]);
    if (mesh_block == nullptr) {
      continue;
    }

    // Convert the actual block.
    meshBlockMessageFromMeshBlock(*mesh_block, &mesh_msg->blocks[i]);
  }

  for (const Index3D & block_index : block_indices_to_delete) {
    mesh_msg->block_indices.push_back(index3DMessageFromIndex3D(block_index));
    mesh_msg->blocks.push_back(nvblox_msgs::msg::MeshBlock());
  }
}

void markerMessageFromMeshBlock(const MeshBlock::ConstPtr& mesh_block,
                                const std::string& frame_id,
                                visualization_msgs::msg::Marker* marker,
                                bool minimal_msg) {
  marker->header.frame_id = frame_id;
  marker->ns = "mesh";
  marker->pose.orientation.x = 0;
  marker->pose.orientation.y = 0;
  marker->pose.orientation.z = 0;
  marker->pose.orientation.w = 1;
  if (minimal_msg) {
    marker->scale.x = 0.1;
    marker->scale.y = 0.1;
    marker->scale.z = 0.1;

    // only publish the points
    marker->type = visualization_msgs::msg::Marker::POINTS;
    marker->color.r = 1.0;
    marker->color.a = 1.0;

    std::vector<Vector3f> vertices = mesh_block->getVertexVectorOnCPU();
    marker->points.reserve(vertices.size());
    for (size_t i = 0; i < vertices.size(); i++) {
      marker->points.push_back(pointMessageFromVector(vertices[i]));
    }

  } else {
    marker->scale.x = 1;
    marker->scale.y = 1;
    marker->scale.z = 1;
    marker->type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
    // Assumes UNWELDED mesh: all vertices in order.
    std::vector<Vector3f> vertices = mesh_block->getVertexVectorOnCPU();
    std::vector<Color> colors = mesh_block->getColorVectorOnCPU();
    std::vector<int> triangles = mesh_block->getTriangleVectorOnCPU();

    CHECK_EQ(vertices.size(), colors.size());

    marker->points.reserve(triangles.size());
    marker->colors.reserve(triangles.size());

    for (size_t i = 0; i < triangles.size(); i++) {
      int index = triangles[i];
      if (index >= colors.size() || index >= vertices.size()) {
        continue;
      }
      marker->points.push_back(pointMessageFromVector(vertices[index]));
      marker->colors.push_back(colorMessageFromColor(colors[index]));
    }
  }
}

// Convert a mesh to a marker array.
void markerMessageFromMeshLayer(
    const BlockLayer<MeshBlock>& mesh_layer, const std::string& frame_id,
    visualization_msgs::msg::MarkerArray* marker_msg, bool minimal_msg) {
  // Get all the mesh blocks.
  std::vector<Index3D> indices = mesh_layer.getAllBlockIndices();

  marker_msg->markers.resize(indices.size());

  size_t output_index = 0;
  for (size_t i = 0; i < indices.size(); i++) {
    MeshBlock::ConstPtr mesh_block = mesh_layer.getBlockAtIndex(indices[i]);
    if (mesh_block->size() == 0) {
      continue;
    }
    markerMessageFromMeshBlock(mesh_block, frame_id,
                               &marker_msg->markers[output_index], minimal_msg);
    marker_msg->markers[output_index].id = output_index;
    std::stringstream ns_stream;
    ns_stream << indices[i].x() << "_" << indices[i].y() << "_"
              << indices[i].z();
    marker_msg->markers[output_index].ns = ns_stream.str();
    output_index++;
  }
  marker_msg->markers.resize(output_index);
}

// Convert a mesh to a pointcloud
void pointcloudMessageFromMeshLayer(const BlockLayer<MeshBlock>& mesh_layer,
                                    const std::string& frame_id,
                                    sensor_msgs::msg::PointCloud2* pc_msg, int downsample, bool push_single) {
  timing::Timer timer("ros/mesh/pc");
  
  timing::Timer timer_get_pc("ros/mesh/pc/get_points");
  // first get a list of all the points
  std::vector<Vector3f> vertices;

  // get all blocks
  std::vector<Index3D> indices = mesh_layer.getAllBlockIndices();

  // loop through each block and append the vectors
  for (size_t i = 0; i < indices.size(); i++) {
    MeshBlock::ConstPtr mesh_block = mesh_layer.getBlockAtIndex(indices[i]);
    if (mesh_block->size() == 0) continue;

    // get the list of verticies in the block
    std::vector<Vector3f> block_vertices = mesh_block->getVertexVectorOnCPU();

    // append
    if (push_single)
    {
	    vertices.push_back(block_vertices[0]);
    } else { 
    vertices.insert(vertices.end(), block_vertices.begin(),
                    block_vertices.end());
    }
  }
  timer_get_pc.Stop();
  
  timing::Timer timer_make_msg("ros/mesh/pc/make_msg");

  // now we have all the vertices, we can build the pointcloud

  // fill in the header
  pc_msg->header.frame_id = frame_id;

  // setup the pointcloud datastructure
  pc_msg->height = 1;
  pc_msg->width = vertices.size() / downsample;

  pc_msg->is_dense = true;
  pc_msg->is_bigendian = false;

  // set fields
  sensor_msgs::PointCloud2Modifier modifier(*pc_msg);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  modifier.resize(vertices.size());

  // fill in the pointcloud with data
  sensor_msgs::PointCloud2Iterator<float> iter_x(*pc_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(*pc_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(*pc_msg, "z");

  for (size_t i = 0; i < vertices.size(); i = i + downsample, ++iter_x, ++iter_y, ++iter_z) {
    *iter_x = vertices[i].x();
    *iter_y = vertices[i].y();
    *iter_z = vertices[i].z();
  }
}

}  // namespace conversions
}  // namespace nvblox
