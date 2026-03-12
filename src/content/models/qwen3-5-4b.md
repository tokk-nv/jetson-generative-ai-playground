---
title: "Qwen3.5 4B"
model_id: "qwen3-5-4b"
short_description: "Alibaba's efficient Qwen3.5 4B language model tuned for practical edge deployment"
family: "Alibaba Qwen3.5"
icon: "🔮"
is_new: false
order: 4
type: "Text"
memory_requirements: "4GB RAM"
precision: "AWQ 4-bit"
model_size: "2.5GB"
hf_checkpoint: "cyankiwi/Qwen3.5-4B-AWQ-4bit"
huggingface_url: "https://huggingface.co/Qwen/Qwen3.5-4B"
minimum_jetson: "Orin Nano"
supported_inference_engines:
  - engine: "vLLM"
    type: "Container"
    run_command_orin: "sudo docker run -it --rm --pull always --runtime=nvidia --network host ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin vllm serve cyankiwi/Qwen3.5-4B-AWQ-4bit --gpu-memory-utilization 0.8 --enable-prefix-caching --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder"
    run_command_thor: "sudo docker run -it --rm --pull always --runtime=nvidia --network host ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor vllm serve cyankiwi/Qwen3.5-4B-AWQ-4bit --gpu-memory-utilization 0.8 --enable-prefix-caching --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder"
---

Qwen3.5 4B offers a balanced point in the Qwen3.5 family for local instruction following, coding help, and agent-style workloads while staying practical on smaller Jetson devices.

## Inputs and Outputs

**Input:** Text

**Output:** Text

## Intended Use Cases

- **General assistant**: Everyday instruction-following and chat
- **Code generation**: Lightweight coding and debugging tasks
- **Tool calling**: Structured tool use with vLLM
- **Multilingual tasks**: Translation and multilingual prompting

## Additional Resources

- [Original Model](https://huggingface.co/Qwen/Qwen3.5-4B) - Base Qwen3.5 4B checkpoint
- [AWQ Checkpoint](https://huggingface.co/cyankiwi/Qwen3.5-4B-AWQ-4bit) - Quantized checkpoint used here
