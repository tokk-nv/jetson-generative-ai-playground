---
title: "Matcha Bot - Embodied AI Hackathon 1st Place Winner"
description: "Automating the matcha-making process using two SO-101 robotic arms, NVIDIA Jetson Thor for inference, and the GR00T N1.5 Vision Language Action model."
author: "SIGRobotics UIUC"
date: "2025-10-26"
source: "Hackster"
link: "https://www.hackster.io/sigrobotics/matcha-bot-sigrobotics-embodied-ai-hackathon-1st-place-f0e520"
image: "/images/projects/matcha-bot.png"
featured: true
jetson: ["Jetson Thor"]
tags: ["VLA", "Robotics", "Hackathon"]
---

🏆 **1st Place Winner** at the Embodied AI Hackathon!

Team SIGRobotics from UIUC developed an autonomous matcha-making robot as an MVP for a fully autonomous coffee shop concept.

### Hardware Stack

- **NVIDIA Jetson Thor** - Edge AI inference and processing
- **Modified XLeRobot** - Bimanual SO-101 arm system
- **3 USB 2.0 Cameras** - Multi-perspective vision data

### Software Stack

- **GR00T N1.5** - Vision Language Action Model for decision making
- **LeRobot** - Data collection and teleoperation framework
- **Custom bimanual control packages** - For dual-arm coordination

### Three Core Tasks

1. **Pouring matcha powder** - Precise powder dispensing
2. **Pouring water** - Controlled liquid handling
3. **Whisking** - Consistent mixing motion

### Key Innovations

- Custom LeRobot packages for bimanual SO-101 control
- Low-latency inference system with Jetson Thor as server
- Clever state encoding to handle model memory limitations
- Web interface prototype for voice-controlled operation

### Team Members

- Aarsh Mittal
- Keshav Badrinath  
- Himank Handa
- Leo L (LeoTheBub)

### Resources

- [Hackster Project Page](https://www.hackster.io/sigrobotics/matcha-bot-sigrobotics-embodied-ai-hackathon-1st-place-f0e520)
- [Isaac GR00T N1.5 Repository](https://github.com/NVIDIA/Isaac-GR00T)
- [LeRobot Framework](https://github.com/huggingface/lerobot)

