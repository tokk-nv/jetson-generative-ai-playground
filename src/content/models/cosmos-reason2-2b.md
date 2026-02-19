---
title: "Cosmos Reason 2 2B"
model_id: "cosmos-reason2-2b"
short_description: "NVIDIA's compact 2B parameter vision-language model with built-in chain-of-thought reasoning for edge deployment"
family: "NVIDIA Cosmos Reason"
icon: "🧠"
is_new: true
order: 2
type: "Multimodal"
memory_requirements: "8GB RAM"
precision: "FP8"
model_size: "5GB"
hf_checkpoint: "nvidia/Cosmos-Reason2-2B"
huggingface_url: "https://huggingface.co/nvidia/Cosmos-Reason2-2B"
minimum_jetson: "Orin Super Nano"
supported_inference_engines:
  - engine: "vLLM"
    type: "Container"
    install_command: "ngc registry model download-version \"nim/nvidia/cosmos-reason2-2b:1208-fp8-static-kv8\""
    run_command_orin: "sudo docker run -it --rm --runtime=nvidia --network host --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 -v $MODEL_PATH:/models/cosmos-reason2-2b:ro ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04 bash -c 'cd /opt && source venv/bin/activate && vllm serve /models/cosmos-reason2-2b --max-model-len 8192 --gpu-memory-utilization 0.8 --reasoning-parser qwen3'"
    run_command_thor: "sudo docker run -it --rm --runtime=nvidia --network host --ipc host -v $MODEL_PATH:/models/cosmos-reason2-2b:ro nvcr.io/nvidia/vllm:26.01-py3 vllm serve /models/cosmos-reason2-2b --max-model-len 8192 --gpu-memory-utilization 0.8 --reasoning-parser qwen3"
---

[NVIDIA Cosmos Reasoning 2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) is a compact vision-language model with built-in chain-of-thought reasoning capabilities. Despite its small 2B parameter size, it can perform spatial reasoning, anomaly detection, and detailed scene analysis, making it well-suited for edge deployment on Jetson.

> **Note:** The HuggingFace version of this model does not fit on Orin Nano. You must download the **[FP8 quantized checkpoint from NGC](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/models/cosmos-reason2-2b/files?version=1208-fp8-static-kv8)** and mount it as a volume. Set `MODEL_PATH` to the downloaded checkpoint directory before running.

## Key Capabilities

- **Spatial Reasoning**: Understands spatial relationships between objects in scenes
- **Anomaly Detection**: Identifies unusual patterns or objects in visual data
- **Scene Analysis**: Provides detailed descriptions and analysis of visual content
- **Chain-of-thought Reasoning**: Generates reasoning traces before concluding with a final response

## Platform Support

| | Jetson AGX Thor | Jetson AGX Orin | Orin Super Nano |
|---|---|---|---|
| **vLLM Container** | `nvcr.io/nvidia/vllm:26.01-py3` | `ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04` | `ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04` |
| **Model** | FP8 via NGC (volume mount) | FP8 via NGC (volume mount) | FP8 via NGC (volume mount) |
| **Max Model Length** | 8192 tokens | 8192 tokens | 768 tokens (memory-constrained) |
| **GPU Memory Util** | 0.8 | 0.8 | 0.52 |

## Orin Super Nano Notes

On Orin Super Nano, use memory-constrained flags:

```bash
vllm serve /models/cosmos-reason2-2b \
  --enforce-eager --max-model-len 768 \
  --max-num-batched-tokens 768 \
  --gpu-memory-utilization 0.52 \
  --max-num-seqs 1 --enable-chunked-prefill \
  --limit-mm-per-prompt '{"image":1}'
```

You may also need to reduce the image resolution in `preprocessor_config.json`.

## Inputs and Outputs

**Input:**
- Text prompts and images
- Supports video frame analysis via `--media-io-kwargs`

**Output:**
- Generated text with chain-of-thought reasoning traces
- Spatial analysis, anomaly detection results, and scene descriptions

## Additional Resources

- [NGC FP8 Checkpoint](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/models/cosmos-reason2-2b/files?version=1208-fp8-static-kv8) — FP8 quantized model for Jetson deployment
- [Live VLM WebUI](https://github.com/NVIDIA-AI-IOT/live-vlm-webui) — real-time webcam-to-VLM interface
