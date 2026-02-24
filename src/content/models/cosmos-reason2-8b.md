---
title: "Cosmos Reason 2 8B"
model_id: "cosmos-reason2-8b"
short_description: "NVIDIA's 8B parameter vision-language model with advanced chain-of-thought reasoning capabilities"
family: "NVIDIA Cosmos Reason"
icon: "🧠"
is_new: true
order: 3
type: "Multimodal"
memory_requirements: "18GB RAM"
precision: "FP8"
model_size: "10GB"
hf_checkpoint: "nvidia/Cosmos-Reason2-8B"
huggingface_url: "https://huggingface.co/nvidia/Cosmos-Reason2-8B"
minimum_jetson: "AGX Orin"
supported_inference_engines:
  - engine: "llama.cpp"
    type: "Container"
    run_command_orin: "sudo docker run -it --rm --pull always --runtime=nvidia --network host -v $HOME/.cache/huggingface:/root/.cache/huggingface ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin llama-server -hf Kbenkhaled/Cosmos-Reason2-8B-GGUF:Q8_0"
    run_command_thor: "sudo docker run -it --rm --pull always --runtime=nvidia --network host -v $HOME/.cache/huggingface:/root/.cache/huggingface ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-thor llama-server -hf Kbenkhaled/Cosmos-Reason2-8B-GGUF:Q8_0"
  - engine: "vLLM"
    type: "Container"
    install_command: "ngc registry model download-version \"nim/nvidia/cosmos-reason2-8b:1208-fp8-static-kv8\""
    run_command_orin: "sudo docker run -it --rm --runtime=nvidia --network host --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 -v $MODEL_PATH:/models/cosmos-reason2-8b:ro ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04 bash -c 'cd /opt && source venv/bin/activate && vllm serve /models/cosmos-reason2-8b --max-model-len 8192 --gpu-memory-utilization 0.8 --reasoning-parser qwen3'"
    run_command_thor: "sudo docker run -it --rm --runtime=nvidia --network host --ipc host -v $MODEL_PATH:/models/cosmos-reason2-8b:ro nvcr.io/nvidia/vllm:26.01-py3 vllm serve /models/cosmos-reason2-8b --max-model-len 8192 --gpu-memory-utilization 0.8 --reasoning-parser qwen3"
---

[NVIDIA Cosmos Reason 2 8B](https://huggingface.co/nvidia/Cosmos-Reason2-8B) is the larger variant in the Cosmos Reason 2 family, offering enhanced reasoning performance with 8 billion parameters. It provides stronger chain-of-thought reasoning capabilities compared to the 2B variant, suitable for more demanding vision-language tasks on Jetson.

## Key Capabilities

- **Enhanced Reasoning**: Stronger chain-of-thought reasoning compared to the 2B variant
- **Spatial Reasoning**: Advanced understanding of spatial relationships between objects
- **Anomaly Detection**: Identifies unusual patterns and anomalies in visual data
- **Scene Analysis**: Comprehensive and detailed analysis of complex visual scenes
- **Video Understanding**: Supports video frame analysis for temporal reasoning

## Running with llama.cpp

GGUF models are downloaded automatically:

```bash
sudo docker run -it --rm --pull always --runtime=nvidia --network host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin \
  llama-server -hf Kbenkhaled/Cosmos-Reason2-8B-GGUF:Q8_0
```

Available quantizations: Q8_0 (8.2 GB, best quality), Q4_K_M (4.7 GB, smaller footprint), F16 (16 GB, full precision). The built-in web UI is at `http://localhost:8080` with an OpenAI-compatible API on the same port.

## Running with vLLM (AGX Orin / Thor)

The vLLM path uses an [FP8 quantized checkpoint from NGC](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/models/cosmos-reason2-8b?version=1208-fp8-static-kv8) downloaded via the NGC CLI.

### Step 1: Install and Configure the NGC CLI

```bash
wget -O ngccli_arm64.zip https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/4.13.0/files/ngccli_arm64.zip
unzip ngccli_arm64.zip && chmod u+x ngc-cli/ngc
export PATH="$PATH:$(pwd)/ngc-cli"
ngc config set
```

### Step 2: Download the FP8 Model

```bash
ngc registry model download-version "nim/nvidia/cosmos-reason2-8b:1208-fp8-static-kv8"
MODEL_PATH="$(pwd)/cosmos-reason2-8b_v1208-fp8-static-kv8"
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

> **Note:** Due to the 8B model size, this model requires at least Jetson AGX Orin with 64GB or Jetson AGX Thor.

## Cosmos Reason 2 Family

| Model | Parameters | Memory | Best For |
|---|---|---|---|
| [Cosmos Reason 2 2B](/models/cosmos-reason2-2b) | 2B | 8GB RAM | Edge deployment, Orin Nano |
| **Cosmos Reason 2 8B** | 8B | 18GB RAM | Higher accuracy, AGX Orin / Thor |

## Additional Resources

- [GGUF Quantizations](https://huggingface.co/Kbenkhaled/Cosmos-Reason2-8B-GGUF) - Q8_0, Q4_K_M, and F16 for llama.cpp
- [NGC FP8 Checkpoint](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/models/cosmos-reason2-8b?version=1208-fp8-static-kv8) - FP8 quantized model for vLLM
- [Live VLM WebUI](https://github.com/NVIDIA-AI-IOT/live-vlm-webui) - real-time webcam-to-VLM interface
