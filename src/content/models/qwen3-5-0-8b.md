---
title: "Qwen3.5 0.8B"
model_id: "qwen3-5-0-8b"
short_description: "Alibaba's compact Qwen3.5 language model for lightweight edge deployment"
family: "Alibaba Qwen3.5"
icon: "🔮"
is_new: false
order: 5
type: "Text"
memory_requirements: "2GB RAM"
precision: "BF16"
model_size: "1.7GB"
hf_checkpoint: "Qwen/Qwen3.5-0.8B"
huggingface_url: "https://huggingface.co/Qwen/Qwen3.5-0.8B"
minimum_jetson: "Orin Nano"
supported_inference_engines:
  - engine: "vLLM"
    type: "Container"
    run_command_orin: "sudo docker run -it --rm --pull always --runtime=nvidia --network host ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin vllm serve Qwen/Qwen3.5-0.8B --gpu-memory-utilization 0.8 --enable-prefix-caching --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder"
    run_command_thor: "sudo docker run -it --rm --pull always --runtime=nvidia --network host ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor vllm serve Qwen/Qwen3.5-0.8B --gpu-memory-utilization 0.8 --enable-prefix-caching --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder"
---

Qwen3.5 0.8B is the smallest text model in the Qwen3.5 lineup. It is designed for lightweight local inference, fast iteration, and low-memory deployments on Jetson.

## Inputs and Outputs

**Input:** Text

**Output:** Text

## Intended Use Cases

- **Lightweight chat**: Low-latency conversational inference
- **Edge automation**: Small assistants and embedded agents
- **Tool calling**: OpenAI-compatible tool use via vLLM
- **Rapid prototyping**: Quick local experiments on constrained devices

## Additional Resources

- [Hugging Face Model](https://huggingface.co/Qwen/Qwen3.5-0.8B) - Original checkpoint
