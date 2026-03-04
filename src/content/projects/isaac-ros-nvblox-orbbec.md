---
title: "Isaac-ROS NVBlox with Orbbec RGB-D Camera on AGX Orin"
description: "Deploy Isaac-ROS NVBlox for 3D reconstruction and obstacle mapping using an Orbbec RGB-D camera on Jetson AGX Orin with a mobile chassis for autonomous navigation."
author: "Seeed Studio"
date: "2024-08-15"
source: "Seeed"
link: "https://wiki.seeedstudio.com/deploy_nvblox_jetson_agx_orin/"
image: "https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-4.0/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_nvblox/isaac_sim_nvblox_humans.gif"
featured: true
jetson: ["Jetson AGX Orin"]
tags: ["Robotics", "Isaac ROS", "ROS2", "Docker"]
---

This comprehensive guide from Seeed Studio demonstrates how to deploy NVIDIA's Isaac-ROS NVBlox package for real-time 3D reconstruction and obstacle mapping using an Orbbec RGB-D camera on the Jetson AGX Orin platform.

### What is NVBlox?

NVBlox is an NVIDIA Isaac ROS package that performs GPU-accelerated 3D scene reconstruction and generates a 2D costmap for navigation. It's designed for autonomous mobile robots (AMRs) and provides:

- Real-time 3D mesh reconstruction
- 2D costmap generation for obstacle avoidance
- GPU-accelerated TSDF (Truncated Signed Distance Function) fusion
- Integration with ROS 2 navigation stack

### Hardware Setup

The project utilizes:
- **NVIDIA Jetson AGX Orin** - Edge AI computing platform
- **Orbbec RGB-D Camera** - Depth sensing camera (Gemini 2 or similar)
- **Mobile Chassis** - Wheeled robot base for autonomous navigation
- **reComputer J4012** - Seeed's Jetson-based industrial computer

### Key Features

- Step-by-step deployment instructions
- Docker container setup for Isaac ROS
- Camera calibration and configuration
- RViz visualization integration
- Mobile robot integration examples

### Use Cases

- Autonomous mobile robot navigation
- Indoor 3D mapping
- Obstacle detection and avoidance
- Warehouse and logistics automation
- Research and prototyping

### Resources

- [Seeed Studio Wiki Guide](https://wiki.seeedstudio.com/deploy_nvblox_jetson_agx_orin/)
- [NVIDIA Isaac ROS NVBlox Documentation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_nvblox/index.html)
- [Orbbec SDK](https://github.com/orbbec/OrbbecSDK)

