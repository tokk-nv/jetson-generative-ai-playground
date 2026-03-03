---
title: "OpenClaw: Private AI Assistant on Jetson Thor"
description: "Run OpenClaw (180K+ GitHub stars) as a fully local AI assistant on Jetson Thor using Docker Model Runner. Supports 8B–30B parameter models with Telegram, WhatsApp, and Discord integration — no cloud, no API costs."
author: "Ajeet Singh Raina"
date: "2026-02-13"
source: "Other"
link: "https://www.ajeetraina.com/how-to-run-openclaw-moltbot-on-nvidia-jetson-thor-with-docker-model-runner-your-private-ai-assistant-at-the-edge/"
image: "https://www.ajeetraina.com/content/images/size/w2000/2026/02/Screenshot-2026-02-14-at-23.35.49.png"
featured: false
jetson: ["Jetson Thor"]
tags: ["LLM", "Agentic AI", "Docker"]
---

Ajeet Singh Raina demonstrates how to deploy OpenClaw — the open-source AI assistant with 180K+ GitHub stars — on Jetson AGX Thor using Docker Model Runner for fully local inference. The setup runs 8B–30B parameter models (Qwen3, Llama 3.2, Qwen3 Coder 30B MoE) with zero API costs and full privacy.

The guide covers the complete stack: Docker Model Runner with CUDA acceleration, OpenClaw gateway configuration, Telegram bot integration, and a Docker Compose setup for reproducible deployment. Thor's 128 GB unified memory enables running frontier-class models that typically require data center GPUs.

### Key Technical Details

- Docker Model Runner with CUDA backend on Jetson Thor
- Qwen3 8B at ~12–15 tok/s, Llama 3.2 3B at ~17 tok/s
- 64K context window for complex agentic workflows
- Multi-channel support: Telegram, WhatsApp, Discord, Slack
- Full Docker Compose stack for one-command deployment

### Resources

- [Full Tutorial](https://www.ajeetraina.com/how-to-run-openclaw-moltbot-on-nvidia-jetson-thor-with-docker-model-runner-your-private-ai-assistant-at-the-edge/)
