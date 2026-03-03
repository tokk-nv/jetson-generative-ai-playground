---
title: "Sprout: AI-Powered Autonomous Microfarm on Jetson Orin Nano"
description: "TreeHacks 2026 Edge AI track 3rd place winner. An autonomous robotic garden powered by Jetson Orin Nano that uses on-device vision AI to identify plant species, assess health, and deliver precision watering through a custom-built gantry system."
author: "Eric, Gavin, Ethan, Markus "
date: "2026-02-14"
source: "Hackster"
link: "https://devpost.com/software/greenguardian-microfarm"
video: "https://www.youtube.com/embed/PbnM8TVgxBQ"
image: "https://img.youtube.com/vi/PbnM8TVgxBQ/maxresdefault.jpg"
featured: true
jetson: ["Jetson Orin Nano"]
tags: ["Robotics", "IoT", "VLM", "Hackathon"]
---

Sprout placed 3rd in the Edge AI track at TreeHacks 2026. It's an autonomous robotic garden that uses a Jetson Orin Nano with a gantry-mounted camera to monitor plants and deliver precision watering — all running entirely on-device.

A custom edge AI model running on the Jetson Orin Nano analyzes live camera footage to identify plant species and assess health status in real-time. Based on the plant type and condition, the system calculates exactly how much water is needed and actuates the pump accordingly.

The team built the 3-axis gantry system from scratch using machined sheet metal components, custom motor driver circuits, and 3D-printed parts — combining mechanical engineering, electrical design, and edge AI into a single weekend build.

### Key Technical Details

- Jetson Orin Nano running custom plant classification and health assessment model
- 3-axis CNC-style gantry with precision coordinate mapping
- Custom motor driver circuits and pump switching circuits
- Fully offline inference — no cloud dependency
- Arduino-based low-level hardware control with Jetson AI pipeline

### Resources

- [Devpost Project](https://devpost.com/software/greenguardian-microfarm)
