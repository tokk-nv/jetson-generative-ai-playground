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
minimum_jetson: "Orin Nano"
supported_inference_engines:
  - engine: "llama.cpp"
    type: "Container"
    run_command_orin: "sudo docker run -it --rm --pull always --runtime=nvidia --network host -v $HOME/.cache/huggingface:/root/.cache/huggingface ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin llama-server -hf Kbenkhaled/Cosmos-Reason2-2B-GGUF:Q8_0"
    run_command_thor: "sudo docker run -it --rm --pull always --runtime=nvidia --network host -v $HOME/.cache/huggingface:/root/.cache/huggingface ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-thor llama-server -hf Kbenkhaled/Cosmos-Reason2-2B-GGUF:Q8_0"
  - engine: "vLLM"
    type: "Container"
    install_command: "ngc registry model download-version \"nim/nvidia/cosmos-reason2-2b:1208-fp8-static-kv8\""
    run_command_orin: "sudo docker run -it --rm --runtime=nvidia --network host --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 -v $MODEL_PATH:/models/cosmos-reason2-2b:ro ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04 bash -c 'cd /opt && source venv/bin/activate && vllm serve /models/cosmos-reason2-2b --max-model-len 8192 --gpu-memory-utilization 0.8 --reasoning-parser qwen3'"
    run_command_thor: "sudo docker run -it --rm --runtime=nvidia --network host --ipc host -v $MODEL_PATH:/models/cosmos-reason2-2b:ro nvcr.io/nvidia/vllm:26.01-py3 vllm serve /models/cosmos-reason2-2b --max-model-len 8192 --gpu-memory-utilization 0.8 --reasoning-parser qwen3"
---

[NVIDIA Cosmos Reasoning 2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) is a compact vision-language model with built-in chain-of-thought reasoning capabilities. Despite its small 2B parameter size, it can perform spatial reasoning, anomaly detection, and detailed scene analysis, making it well-suited for edge deployment on Jetson.

## Key Capabilities

- **Spatial Reasoning**: Understands spatial relationships between objects in scenes
- **Anomaly Detection**: Identifies unusual patterns or objects in visual data
- **Scene Analysis**: Provides detailed descriptions and analysis of visual content
- **Chain-of-thought Reasoning**: Generates reasoning traces before concluding with a final response

## Running with llama.cpp (Recommended for Orin Nano)

The fastest way to get started. GGUF models are downloaded automatically:

```bash
sudo docker run -it --rm --pull always --runtime=nvidia --network host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin \
  llama-server -hf Kbenkhaled/Cosmos-Reason2-2B-GGUF:Q8_0
```

Available quantizations: Q8_0 (2.1 GB, best quality), Q4_K_M (1.2 GB, smaller footprint). The built-in web UI is at `http://localhost:8080` with an OpenAI-compatible API on the same port.

## Running with vLLM (AGX Orin / Thor)

The vLLM path uses an [FP8 quantized checkpoint from NGC](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/models/cosmos-reason2-2b/files?version=1208-fp8-static-kv8) downloaded via the NGC CLI.

### Step 1: Install and Configure the NGC CLI

```bash
wget -O ngccli_arm64.zip https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/4.13.0/files/ngccli_arm64.zip
unzip ngccli_arm64.zip && chmod u+x ngc-cli/ngc
export PATH="$PATH:$(pwd)/ngc-cli"
ngc config set
```

### Step 2: Download the FP8 Model

```bash
ngc registry model download-version "nim/nvidia/cosmos-reason2-2b:1208-fp8-static-kv8"
MODEL_PATH="$(pwd)/cosmos-reason2-2b_v1208-fp8-static-kv8"
```

### Step 3: Serve

```bash
sudo sysctl -w vm.drop_caches=3
```

| | Jetson AGX Thor | Jetson AGX Orin (64GB) |
|---|---|---|
| **vLLM Container** | `nvcr.io/nvidia/vllm:26.01-py3` | `ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04` |
| **Max Model Length** | 8192 tokens | 8192 tokens |
| **GPU Memory Util** | 0.8 | 0.8 |

## Inputs and Outputs

**Input:** Text prompts and images, video frame analysis

**Output:** Generated text with chain-of-thought reasoning traces, spatial analysis, anomaly detection, and scene descriptions

## Additional Resources

- [GGUF Quantizations](https://huggingface.co/Kbenkhaled/Cosmos-Reason2-2B-GGUF) - Q8_0, Q4_K_M, and F16 for llama.cpp
- [NGC FP8 Checkpoint](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/models/cosmos-reason2-2b/files?version=1208-fp8-static-kv8) - FP8 quantized model for vLLM
- [Live VLM WebUI](https://github.com/NVIDIA-AI-IOT/live-vlm-webui) - real-time webcam-to-VLM interface
