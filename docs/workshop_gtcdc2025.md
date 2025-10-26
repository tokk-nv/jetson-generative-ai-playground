# From AI Exploration to Production Deployment

![](./images/jetson-agx-thor-family-key-visual-03-v002-eb-1k.jpg){ width="40%"  align=right}

*Master inference optimization on Jetson Thor with vLLM*

> Welcome! In this hands-on workshop, you’ll unlock truly high-performance, **on-device** generative AI using the new [**NVIDIA Jetson Thor**](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/). You’ll start with a clean quality baseline, then step through practical optimizations -- **FP8**, **FP4**, and **speculative decoding** -- measuring speed vs. quality at each stage.

## What you’ll build

- A local **vLLM** service running on Jetson Thor
- A simple **OpenAI-compatible** chat endpoint
- An **Open WebUI** front-end bound to your local vLLM server
- A repeatable procedure to compare FP16 → FP8 → FP4 → FP4+SpecDec

!!! tip "Who is this for?"
    Teams building edge applications/products (robots, kiosks, appliances) who need **fast, private, API-compatible** LLMs without cloud dependency.

!!! info "🌸 GTC DC 2025 Workshop Setup"
    **Welcome to the GTC DC 2025 "Accelerate Generative AI Inference on Jetson Thor" Workshop!**

    ### What We Provide:
    - **Hardware**: Jetson Thor rack setup with mini LCD display
    - **Network**: Local workshop network (see topology diagram below)
    - **Pre-installed**: BSP preinstalled and set up, VLLM container pulled and ready
    - **Support**: Technical Assistants (TAs) available throughout the session
    - **Jetson HUD**: Utility for LCD display monitoring

    ### Network Topology:
    [Add your network diagram here]

    ### Jetson HUD:
    [Add your image here]

    ### Need Help?
    **Use the "Red-Cup, Blue-Cup" system:**

    - 🔴 **Red cup on top**: I need help from a TA
    - 🔵 **Blue cup on top**: I'm good to go (problem resolved)

---

## Prerequisites

- **Hardware**: Jetson **Thor** devkit (128 GB unified memory recommended)
- **Software**: JetPack 6.x+ (Thor image), Docker (rootless or root OK)
- **Network**: Internet access to pull containers/models (or a pre-warmed registry/cache)
- **CLI**: `docker`, `wget`, `curl`, `python3` (optional for quick tests)

??? info "Why Thor?"
    Thor’s memory capacity enables **large models** and **large context windows**, allows **serving multiple models concurrently**, and supports **high-concurrency batching** on-device.

---

## 🚀 Experience: Thor's Raw Power with 120B Intelligence

### The Open Weights Revolution 🔓

**What are Open Weights Models?**<br>
Unlike **closed models** (GPT-4, Claude, Gemini), **open weights models** give you:

- **Complete model access**: Download and run locally
- **Data privacy**: Your data never leaves your device
- **No API dependencies**: Work offline, no rate limits
- **Customization freedom**: Fine-tune for your specific needs
- **Cost control**: No per-token charges

### Why This Matters: Closed vs. Open Comparison

| Aspect | Closed Models (GPT-4, etc.) | Open Weights Models |
|--------|------------------------------|---------------------|
| **Privacy** | Data sent to external servers | Stays on your device |
| **Latency** | Network dependent | Local inference speed |
| **Availability** | Internet required | Works offline |
| **Customization** | Limited via prompts | Full fine-tuning possible |
| **Cost** | Pay per token/request | Hardware cost only |
| **Compliance** | External data handling | Full control |

### Enter GPT-OSS-120B: Game Changer 🎯

**OpenAI's GPT-OSS-120B** represents a breakthrough:

- **First major open weights model** from OpenAI
- **120 billion parameters** of GPT-quality intelligence
- **Massive compute requirements** - needs serious hardware

**The Thor Advantage:**

- **One of the few platforms** capable of running GPT-OSS-120B at the edge
- **Real-time inference** without cloud dependencies
- **Perfect for evaluation**: Test if the model fits your domain
- **Baseline assessment**: Understand capabilities before fine-tuning

### Why Start Here?

Before you invest in fine-tuning or domain adaptation:

1. **Domain Knowledge Check**: Does the base model understand your field?
2. **Performance Baseline**: How well does it perform out-of-the-box?
3. **Use Case Validation**: Is this the right model architecture?
4. **Resource Planning**: What hardware do you actually need?

**Thor lets you answer these questions locally, privately, and immediately.**


### Jetson set-up

??? note "Jetson set up (GTC DC workshop attendees get to skip)"

    You want to do

    - BSP installation
      - https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/quick_start.html
    - Docker setup https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_docker.html

??? note "Pre-Workshop Setup (For Organizers)"

    !!! info "Directory Structure Alignment"
        **Following jetson-containers convention with optimization caching:**

        We use `$ROOT/data/` as the unified data directory structure:
        ```
        $ROOT/data/
        ├── models/
        │   └── huggingface/          # Model cache (122GB)
        └── vllm_cache/               # Torch compile cache (~2GB)
            └── torch_compile_cache/
        ```

        This ensures:

        - ✅ **Consistency** with existing Jetson workflows
        - ✅ **Familiar paths** for jetson-containers users
        - ✅ **Easy integration** with other Jetson AI tools
        - ✅ **Standardized model storage** across projects
        - ✅ **Pre-warmed optimization** for instant startup

    !!! warning "Storage Requirements"
        **Total storage needed per Thor unit:**
        - **GPT-OSS-120B model**: ~122GB
        - **vLLM compilation cache**: ~2GB
        - **Container images**: ~10GB
        - **Workspace**: ~5GB

        **Recommended per unit**: 200GB+ free space for comfortable operation

    **Model Pre-download Process:**
    ```bash
    # 1. Download model once (takes 30-60 minutes depending on network)
    sudo docker run --rm -it --runtime=nvidia --name=vllm-download \
      nvcr.io/nvidia/vllm:25.09-py3

    # Inside container, trigger model download:
    python -c "
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('openai/gpt-oss-120b')
    print('Model downloaded successfully!')
    "

    # 2. Copy model cache to host (jetson-containers structure)
    mkdir -p $ROOT/data/models
    docker cp vllm-download:/root/.cache/huggingface $ROOT/data/models/

    # 3. Verify model size
    du -h $ROOT/data/models/huggingface/hub/models--openai--gpt-oss-120b/
    # Should show ~122GB
    ```

    **Distribute to all workshop units:**
    ```bash
    # Copy to each Thor unit (adjust IPs/hostnames)
    for unit in thor-{01..60}; do
      rsync -av --progress $ROOT/data/models/ ${unit}:$ROOT/data/models/
    done
    ```

### Exercise: Let's get working!

```bash
docker run --rm -it \
  --network host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --runtime=nvidia \
  --name=vllm \
  -v $ROOT/data/models/huggingface:/root/.cache/huggingface \
  -v $ROOT/data/vllm_cache:/root/.cache/vllm \
  nvcr.io/nvidia/vllm:25.09-py3
```

**Verify pre-downloaded model:**
```bash
# Inside the container, check if model is available
ls -la /root/.cache/huggingface/hub/models--openai--gpt-oss-120b/
du -h /root/.cache/huggingface/hub/models--openai--gpt-oss-120b/
# Should show ~122GB - no download needed!
```

Trick to make gpt-oss work.

```bash
mkdir /etc/encodings
wget https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken -O /etc/encodings/cl100k_base.tiktoken
wget https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken -O /etc/encodings/o200k_base.tiktoken
export TIKTOKEN_ENCODINGS_BASE=/etc/encodings
vllm serve openai/gpt-oss-120b
```

Actual serving

> Please note that the vllm serve command might take some time, but you’ll know once it’s ready when you see this:
> ```bash
> (APIServer pid=92) INFO:     Waiting for application startup.
> (APIServer pid=92) INFO:     Application startup complete.
```

### Test the API (Optional)

Test your vLLM server is working:

```bash
# Check available models
curl http://localhost:8000/v1/models

# Test chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "Hello! Tell me about Jetson Thor."}],
    "max_tokens": 100
  }'
```

!!! note "Understanding the Serving Architecture"
    **What does "serve" mean?**

    When vLLM "serves" a model, it:

    - ✅ **Loads the model** into GPU memory (GPT-OSS-120B)
    - ✅ **Creates an API server** listening on port 8000
    - ✅ **Exposes HTTP endpoints** for inference requests
    - ✅ **Handles concurrent requests** with optimized batching

    **What is vLLM exposing?**

    vLLM creates a **REST API server** at `http://localhost:8000` with endpoints like:

    - `/v1/chat/completions` - Chat-style conversations
    - `/v1/completions` - Text completion
    - `/v1/models` - List available models
    - `/health` - Server health check

    **OpenAI-Compatible Endpoint**

    vLLM implements the **same API format** as OpenAI's GPT models:

    - ✅ **Same request format** - JSON with `messages`, `model`, `max_tokens`
    - ✅ **Same response format** - Structured JSON responses
    - ✅ **Drop-in replacement** - Existing OpenAI code works unchanged
    - ✅ **Local inference** - No data leaves your device

### Launch Open WebUI

Start the web interface for easy interaction:

```bash
docker run -d \
  --network=host \
  -v ${HOME}/open-webui:/app/backend/data \
  -e OPENAI_API_BASE_URL=http://0.0.0.0:8000/v1 \
  --name open-webui --restart always \
  ghcr.io/open-webui/open-webui:main
```

**Access the interface:**

1. Open your browser to `http://localhost:8080`
2. Create an account (stored locally)
3. Start chatting with your local 120B model!

!!! note "About Open WebUI"
    **What role does Open WebUI play?**

    Open WebUI is a **web-based chat interface** that:

    - 🌐 **Provides a familiar ChatGPT-like UI** in your browser
    - 🔌 **Connects to your local vLLM server** (not OpenAI's servers)
    - 💬 **Handles conversations** with chat history and context
    - 🎛️ **Offers model controls** (temperature, max tokens, etc.)
    - 📊 **Shows performance metrics** (tokens/sec, response time)

    **Architecture Flow:**
    ```
    You → Open WebUI (Browser) → vLLM Server → GPT-OSS-120B → Response
    ```

    **Key Benefits:**

    - 🔒 **Complete privacy** - No data sent to external servers
    - ⚡ **Local performance** - Thor's inference speed
    - 🎯 **Production testing** - Real application interface
    - 📈 **Performance monitoring** - See actual tokens/sec


### Troubleshooting vLLM Launch Issues

!!! warning "Common Issue: NVML Errors and Model Architecture Failures"
    If you see errors like:
    - `Can't initialize NVML`
    - `NVMLError_Unknown: Unknown Error`
    - `Model architectures ['GptOssForCausalLM'] failed to be inspected`

    **Root Cause**: Malformed Docker daemon configuration

**Check Docker daemon.json:**
```bash
cat /etc/docker/daemon.json
```

**If the file is missing the default-runtime configuration:**
```json
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
// ❌ Missing "default-runtime": "nvidia" !
```

**Fix with complete configuration:**
```bash
sudo nano /etc/docker/daemon.json
```

**Correct content:**
```json
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "default-runtime": "nvidia"
}
```

**Apply the fix:**
```bash
# Restart Docker daemon
sudo systemctl restart docker

# Verify Docker is running
sudo systemctl status docker

# Test NVIDIA runtime (Thor-compatible CUDA 13)
docker run --rm --runtime=nvidia nvcr.io/nvidia/cuda:13.0.0-runtime-ubuntu24.04 nvidia-smi

# Restart your vLLM container
docker stop vllm  # if running
# Then relaunch with the corrected Docker configuration
```

!!! tip "If Docker Runtime Issues Persist"
    **Try a system reboot** - this often resolves Docker runtime configuration issues:

    ```bash
    sudo reboot
    ```

    **Why reboot helps:**
    - Complete Docker daemon restart with new configuration
    - Fresh NVIDIA driver/runtime initialization
    - Proper CDI device registration
    - All system services start in correct order

    **After reboot, test immediately:**
    ```bash
    docker run --rm --runtime=nvidia nvcr.io/nvidia/cuda:13.0.0-runtime-ubuntu24.04 nvidia-smi
    ```

    **Tested on:** L4T (Jetson Linux) r38.2.2

### GPU Memory Management Issues

!!! warning "Stuck GPU Memory Allocations"
    **Symptom:** vLLM fails with "insufficient GPU memory" despite stopping containers

    **Example error:**
    ```
    ValueError: Free memory on device (14.45/122.82 GiB) on startup is less than
    desired GPU memory utilization (0.7, 85.98 GiB)
    ```

**Diagnosis:**
```bash
# Check current GPU memory usage
jtop
# Expected baseline: ~3-6GB system usage
# Problem: 25GB+ or 100GB+ unexplained usage
```

**Solution sequence:**
```bash
# 1. Stop all containers
docker stop $(docker ps -q) 2>/dev/null
docker rm $(docker ps -aq) 2>/dev/null

# 2. Restart Docker daemon
sudo systemctl restart docker

# 3. Check if memory cleared
jtop

# 4. If memory still high (>10GB baseline), reboot system
sudo reboot
```

!!! info "Root Cause Investigation"
    This appears to be related to GPU memory allocations not being properly released at the driver level. We're investigating the exact cause and will update this section with a more targeted solution.

    **Workaround for now:** System reboot reliably clears stuck allocations.

### What to Expect: Successful vLLM Startup

When everything works correctly, you should see output like this:

```
(APIServer pid=161) INFO: Resolved architecture: GptOssForCausalLM
Parse safetensors files: 100%|████████████████████| 15/15 [00:01<00:00, 14.70it/s]
(APIServer pid=161) INFO: Using max model len 131072
(APIServer pid=161) WARNING: mxfp4 quantization is not fully optimized yet...
Loading safetensors checkpoint shards: 100%|████████████████████| 15/15 [00:24<00:00,  1.64s/it]
INFO: Loading weights took 24.72 seconds
```

**Key indicators of success:**

- ✅ **Architecture resolved**: `GptOssForCausalLM`
- ✅ **Fast loading**: ~25 seconds (vs. hours without pre-cache!)
- ✅ **Quantization active**: `mxfp4` (FP4) for optimal performance
- ✅ **No NVML errors**: GPU acceleration working properly

!!! success "Workshop Magic: Pre-cached Models"
    The **24-second loading time** demonstrates the power of our pre-workshop setup! Without the pre-downloaded model cache, attendees would wait hours for the 122GB download. This is the Thor advantage in action! 🚀

### Startup Time Optimization for Workshops

**Expected Timing Breakdown:**

- ✅ **Model Loading**: ~35 seconds (excellent with pre-cache)
- ⚠️ **Torch Compile**: ~45 seconds (first-time compilation)
- ⚠️ **CUDA Graphs**: ~21 seconds (optimization setup)
- 🏁 **Total**: ~150 seconds (2.5 minutes to full readiness)

**The torch.compile bottleneck** is a one-time cost that creates optimized inference kernels. For workshops, we have two strategies:

!!! tip "Strategy 1: Pre-warm Compilation Cache (Recommended)"
    **For organizers**: Generate compilation cache once, distribute to all units:

    ```bash
    # 1. Pre-workshop: Generate compilation cache
    docker run --rm -it --runtime=nvidia --name=vllm-warmup \
      -v $ROOT/data/models/huggingface:/root/.cache/huggingface \
      -v $ROOT/data/vllm_cache:/root/.cache/vllm \
      nvcr.io/nvidia/vllm:25.09-py3

    # Inside container: Start server once to generate cache
    vllm serve openai/gpt-oss-120b
    # Wait for "Application startup complete", then Ctrl+C

    # 2. Distribute cache to all workshop units
    for unit in thor-{01..60}; do
      rsync -av $ROOT/data/vllm_cache/ ${unit}:$ROOT/data/vllm_cache/
    done

    # 3. Workshop containers use pre-warmed cache
    docker run --rm -it --runtime=nvidia --name=vllm \
      -v $ROOT/data/models/huggingface:/root/.cache/huggingface \
      -v $ROOT/data/vllm_cache:/root/.cache/vllm \
      nvcr.io/nvidia/vllm:25.09-py3
    ```

!!! info "Strategy 2: Fast Startup Mode (Demo-friendly)"
    **For quick demos**: Reduce optimization for faster startup:

    ```bash
    # Faster startup with reduced optimization (~60s total)
    vllm serve openai/gpt-oss-120b \
      --compilation-config '{"level": 0}' \
      --disable-custom-all-reduce
    ```

**Other troubleshooting steps:**

1. **Verify user permissions:**
   ```bash
   groups $USER  # Should include 'docker'
   ```

2. **Check GPU accessibility:**
   ```bash
   nvidia-smi
   # Or test via Docker:
   docker run --rm --runtime=nvidia nvcr.io/nvidia/cuda:13.0.0-runtime-ubuntu24.04 nvidia-smi
   ```

3. **Try alternative model names:**
   ```bash
   vllm serve microsoft/gpt-oss-120b  # Alternative naming
   ```

4. **Use conservative settings:**
   ```bash
   vllm serve openai/gpt-oss-120b \
     --gpu-memory-utilization 0.7 \
     --max-model-len 4096
   ```

Open WebUI

```bash
docker run -d --network=host -v open-webui:/app/backend/data -e OPENAI_API_BASE_URL=http://0.0.0.0:8000/v1 --name open-webui --restart always ghcr.io/open-webui/open-webui:main
```

---

## 🔧 Optimize: Precision Engineering (FP16 → FP8 → FP4)

We’ll use **Llama-3.1-8B-Instruct (FP16)** as our baseline.

### 1.1 Serve baseline model with vLLM

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

Once loaded, select the model in Open WebUI.

### 1.2 Baseline prompt & measurements

**Prompt (copy/paste):**

> *Write a 5-sentence paragraph explaining the main benefit of using Jetson Thor for an autonomous robotics developer.*

**Observe:**

- **Time-to-First-Token (TTFT)** — perceived latency
- **Tokens/sec** — throughput (use Open WebUI stats or API logs)
- **Answer quality** — coherence, accuracy, task fit

??? example "API test without UI"
    ```bash
    curl http://localhost:8000/v1/chat/completions       -H "Content-Type: application/json"       -d '{
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [{"role":"user","content":"Write a 5-sentence paragraph explaining the main benefit of using Jetson Thor for an autonomous robotics developer."}],
            "max_tokens": 256,
            "temperature": 0.7
          }'
    ```

---

### Step 1: "Safe First Step" - Quantize to FP8

FP8 reduces memory bandwidth/footprint and **often matches FP16 quality** for many tasks.

### 2.1 Relaunch in FP8

```bash
vllm serve nvidia/Llama-3.1-8B-Instruct-FP8
```

Select this FP8 variant in Open WebUI and repeat the same prompt.
Compare **TTFT**, **tokens/sec**, and **answer quality** vs. FP16.

---

### Step 2: Push Further - FP4 Quantization

FP4 halves memory again vs. FP8 and is **much faster**, but may introduce noticeable quality drift (hallucinations, repetition).

### 3.1 Relaunch in FP4

```bash
vllm serve nvidia/Llama-3.1-8B-Instruct-FP4
```

Run the **same prompt** and evaluate:

- **Speed**: should be visibly faster than FP16/FP8
- **Quality**: check fidelity to the prompt and coherence

!!! warning "Quality guardrail"
    If you see unacceptable degradation, consider **prompting tweaks**, **temperature/top-p** adjustments, or **domain-specific RAG** to anchor outputs.

---

### Step 3: Advanced Optimization - FP4 + Speculative Decoding (EAGLE3)

Speculative decoding pairs a small **draft model** with your main model. The main model **verifies** multiple drafted tokens in one step—**same final output**, higher throughput.

### 4.1 Relaunch FP4 with speculative decoding

```bash
vllm serve nvidia/Llama-3.1-8B-Instruct-FP4   --trust_remote_code   --speculative-config '{
    "method":"eagle3",
    "model":"yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens":5
  }'
```

Select the FP4 model again in Open WebUI and re-run the **same prompt**.

**Expect:**

- **Throughput**: highest so far
- **Quality**: **identical** to plain FP4 (speculative decoding does not alter the final output)

!!! info "Where this truly shines"
    On **70B-class** models, FP4 + speculative decoding can feel close to smaller models for interactivity, while preserving large-model competence.

---

## Appendix — Operational Notes

### Open WebUI ↔ vLLM bindings

- Default vLLM OpenAI API endpoint: `http://0.0.0.0:8000/v1`
- Ensure WebUI container uses `--network=host` (or map ports accordingly).
- If running behind reverse proxies, pass through **HTTP/1.1 keep-alive** and **WebSocket** if enabled.

### Measuring performance programmatically

- Parse tokens/sec from vLLM logs or use `/v1/completions` with `stream:false` and time deltas.
- Run **3–5 trials** and report **median** for stable comparisons.
- Keep **prompt & sampling params identical** across runs.

### Concurrency & batching

- vLLM’s **PagedAttention** enables high request concurrency.
- Tune **`--max-num-seqs`, `--gpu-memory-utilization`** as your workload grows.
- Use a **fixed set of prompts** to test under load.

---

## Troubleshooting

??? question "Model doesn’t appear in Open WebUI"
    - Confirm WebUI uses `OPENAI_API_BASE_URL=http://<thor-ip>:8000/v1`
    - Verify `docker logs open-webui` shows successful backend registration
    - Check that vLLM is listening on `0.0.0.0:8000`

??? warning "OOM or slow loads"
    - Reduce **context window** or switch to **FP8/FP4**
    - Ensure **swap** is configured appropriately on Thor for your image
    - Close unused sessions/models

??? failure "Tokenizers/encodings error"
    - Re-export `TIKTOKEN_ENCODINGS_BASE`
    - Confirm files exist under `/etc/encodings/*.tiktoken`

---

## What to do next

- Try a **70B** FP4 model with speculative decoding and compare UX to 8B
- Add **RAG** (local vector DB) and measure quality gains vs. baseline
- Evaluate **guardrails** (structured outputs, JSON schemas, tool use)
- Add observability: **latency histograms**, **p95 TTFT**, **tokens/sec**

[Back to Tutorials](../index.md){ .md-button } [Open WebUI how-to](../vit/tutorial_openwebui.md){ .md-button }

---

### TODOs for the lab page (you mentioned you’ll add these)

- [ ] Screenshots/GIFs of Open WebUI model switching
- [ ] A small “Perf panel” screenshot with TTFT/tokens/sec per stage
- [ ] A table summarizing FP16 vs FP8 vs FP4 vs FP4+SpecDec

