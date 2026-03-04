---
title: "TORQ: VLA - Powered Autonomous Driving on Jetson"
description: "TreeHacks 2026 Edge AI track winner. Retrofits a Honda Accord with Jetson AGX Thor running a 10.5B VLA (Alpamayo R1) for high-level reasoning and openpilot for real-time control, with a rider-facing iOS app for transparent AI decision-making."
author: "Subha, Howard, Joel, Bryan "
date: "2026-02-15"
source: "Hackster"
link: "https://devpost.com/software/torq"
video: "https://www.youtube.com/embed/zEDVGfCVhaw"
image: "https://img.youtube.com/vi/zEDVGfCVhaw/maxresdefault.jpg"
featured: true
jetson: ["Jetson Thor"]
tags: ["VLA", "Robotics", "Autonomous Driving", "TensorRT", "Hackathon"]
---

TORQ won the Edge AI track at TreeHacks 2026 by turning a stock 2018 Honda Accord into an autonomous vehicle powered by Jetson AGX Thor. The system fuses two layers of intelligence: NVIDIA's Alpamayo R1 (a 10.5B vision-language-action model) handles high-level scene reasoning — recognizing pedestrians, interpreting lane endings, and planning maneuvers — while a low-level control layer derived from openpilot runs at 100 Hz for smooth steering and braking.

The architecture splits fast reflexes from slow reasoning. The control loop holds the lane and maintains safe following distance independently, so even if the reasoning model stalls, the car stays safe. Cameras connect through the Holoscan Sensor Bridge, routing uncompressed video directly into GPU memory for minimal latency.

A rider-facing iOS app provides real-time transparency: passengers see what the AI sees, why it's making decisions, and can even ask questions or give feedback through natural language — all powered by Alpamayo's language output.

### Key Technical Details

- Alpamayo R1 (10.5B VLA) for high-level driving plans and natural language explanations
- openpilot/sunnypilot fork for 20 Hz vision + 100 Hz control
- Holoscan Sensor Bridge with IMX274 cameras for low-latency video pipeline
- comma.ai Red Panda for CAN bus vehicle control
- MQTT-based communication between Jetson Thor and iOS app

### Resources

- [Devpost Project](https://devpost.com/software/torq)
