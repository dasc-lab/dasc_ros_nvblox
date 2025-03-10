# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import (ComposableNodeContainer, SetParameter,
                                SetParametersFromFile, SetRemap)
from launch_ros.descriptions import ComposableNode


def generate_launch_description():

    bringup_dir = get_package_share_directory('nvblox_examples_bringup')
    base_config_dir = os.path.join(bringup_dir, 'config', 'nvblox')
    specialization_dir = os.path.join(base_config_dir, 'specializations')

    # Config files
    base_config = os.path.join(base_config_dir, 'nvblox_base.yaml')
    humans_config = os.path.join(specialization_dir, 'nvblox_humans.yaml')
    realsense_config = os.path.join(
        specialization_dir, 'nvblox_realsense.yaml')
    simulation_config = os.path.join(
        specialization_dir, 'nvblox_isaac_sim.yaml')

    # Conditionals for setup
    setup_for_isaac_sim = IfCondition(
        LaunchConfiguration('setup_for_isaac_sim', default='False'))
    setup_for_realsense = IfCondition(
        LaunchConfiguration('setup_for_realsense', default='False'))

    # Nvblox node
    node = ComposableNode(
        name='nvblox_human_node',
        package='nvblox_ros',
        plugin='nvblox::NvbloxHumanNode')

    # Nvblox node container
    nvblox_human_container = ComposableNodeContainer(
        name='nvblox_human_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[node],
        output='screen')

    group_action = GroupAction([

        # Set parameters with specializations
        SetParametersFromFile(base_config),
        SetParametersFromFile(humans_config),
        SetParametersFromFile(realsense_config,
                              condition=setup_for_realsense),
        SetParametersFromFile(simulation_config,
                              condition=setup_for_isaac_sim),
        SetParameter(name='global_frame',
                     value=LaunchConfiguration('global_frame', default='odom')),

        # Remappings for realsense data
        SetRemap(src=['depth/image'],
                 dst=['/camera/realsense_splitter_node/output/depth'],
                 condition=setup_for_realsense),
        SetRemap(src=['depth/camera_info'],
                 dst=['/camera/depth/camera_info'],
                 condition=setup_for_realsense),
        SetRemap(src=['color/image'],
                 dst=['/camera/color/image_raw'],
                 condition=setup_for_realsense),
        SetRemap(src=['color/camera_info'],
                 dst=['/camera/color/camera_info'],
                 condition=setup_for_realsense),
        SetRemap(src=['mask/image'],
                 dst=['/unet/raw_segmentation_mask_depadded'],
                 condition=setup_for_realsense),
        SetRemap(src=['mask/camera_info'],
                 dst=['/camera/color/camera_info'],
                 condition=setup_for_realsense),

        # Remappings for isaac sim data
        SetRemap(src=['depth/image'],
                 dst=['/front/stereo_camera/right/depth'],
                 condition=setup_for_isaac_sim),
        SetRemap(src=['depth/camera_info'],
                 dst=['/front/stereo_camera/right/camera_info'],
                 condition=setup_for_isaac_sim),
        SetRemap(src=['color/image'],
                 dst=['/front/stereo_camera/left/rgb'],
                 condition=setup_for_isaac_sim),
        SetRemap(src=['color/camera_info'],
                 dst=['/front/stereo_camera/left/camera_info'],
                 condition=setup_for_isaac_sim),
        SetRemap(src=['mask/image'],
                 dst=['/semantic_conversion/front/stereo_camera/left/semantic_mono8'],
                 condition=setup_for_isaac_sim),
        SetRemap(src=['mask/camera_info'],
                 dst=['/front/stereo_camera/left/camera_info'],
                 condition=setup_for_isaac_sim),
        SetRemap(src=['pointcloud'],
                 dst=['/point_cloud'],
                 condition=setup_for_isaac_sim),

        # Include the node container
        nvblox_human_container
    ])

    return LaunchDescription([group_action])
