---
title: "Cosmos Reason 2 2B"
model_id: "cosmos-reason2-2b"
short_description: "NVIDIA's compact 2B parameter vision-language model with built-in chain-of-thought reasoning for edge deployment"
family: "NVIDIA Cosmos Reason"
icon: "🧠"
is_new: true
hide_run_button: true
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

This model uses an **[FP8 quantized checkpoint from NGC](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/models/cosmos-reason2-2b/files?version=1208-fp8-static-kv8)** which is downloaded via the NGC CLI and mounted into the vLLM container as a volume.

## Key Capabilities

- **Spatial Reasoning**: Understands spatial relationships between objects in scenes
- **Anomaly Detection**: Identifies unusual patterns or objects in visual data
- **Scene Analysis**: Provides detailed descriptions and analysis of visual content
- **Chain-of-thought Reasoning**: Generates reasoning traces before concluding with a final response

## Step 1: Install and Configure the NGC CLI

The [NGC CLI](https://org.ngc.nvidia.com/setup/installers/cli) is needed to download the FP8 model checkpoint from the [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/).

```bash
# Download the NGC CLI for ARM64
wget -O ngccli_arm64.zip https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/4.13.0/files/ngccli_arm64.zip
unzip ngccli_arm64.zip
chmod u+x ngc-cli/ngc
export PATH="$PATH:$(pwd)/ngc-cli"
```

Configure the CLI with your NGC API key (generate one at [NGC API Key setup](https://org.ngc.nvidia.com/setup/api-key)):

```bash
ngc config set
```

You will be prompted for your **API Key**, **CLI output format** (choose `json` or `ascii`), and **org** (press Enter for default).

## Step 2: Download the FP8 Model

```bash
ngc registry model download-version "nim/nvidia/cosmos-reason2-2b:1208-fp8-static-kv8"
```

This creates a directory called `cosmos-reason2-2b_v1208-fp8-static-kv8/`. Set the path for Docker:

```bash
MODEL_PATH="$(pwd)/cosmos-reason2-2b_v1208-fp8-static-kv8"
```

## Step 3: Serve with vLLM

Free cached memory before launching:

```bash
sudo sysctl -w vm.drop_caches=3
```

## Platform Support

| | Jetson AGX Thor | Jetson AGX Orin | Orin Super Nano |
|---|---|---|---|
| **vLLM Container** | `nvcr.io/nvidia/vllm:26.01-py3` | `ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04` | `ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04` |
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

You may also need to reduce the image resolution in `preprocessor_config.json`:

```bash
cd cosmos-reason2-2b_v1208-fp8-static-kv8
cp preprocessor_config.json preprocessor_config.json.bak

python3 -c "
import json
with open('preprocessor_config.json') as f:
    cfg = json.load(f)
cfg['size']['longest_edge'] = 50176
cfg['size']['shortest_edge'] = 3136
with open('preprocessor_config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print('Updated image resolution limits')
"
```

This limits input images to ~50K pixels, keeping image tokens small enough to fit in the constrained context window.

## Inputs and Outputs

**Input:**
- Text prompts and images
- Supports video frame analysis via `--media-io-kwargs`

**Output:**
- Generated text with chain-of-thought reasoning traces
- Spatial analysis, anomaly detection results, and scene descriptions

## Additional Resources

- [NGC FP8 Checkpoint](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/models/cosmos-reason2-2b/files?version=1208-fp8-static-kv8) — FP8 quantized model for Jetson deployment
- [NGC CLI Installers](https://org.ngc.nvidia.com/setup/installers/cli)
- [Live VLM WebUI](https://github.com/NVIDIA-AI-IOT/live-vlm-webui) — real-time webcam-to-VLM interface
