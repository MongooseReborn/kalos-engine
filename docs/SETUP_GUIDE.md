# 🚀 Kalos Engine — Step-by-Step Setup & User Guide

Welcome to the **Kalos Engine Suite** setup guide. This document guides you through selecting any GGUF model size (7B, 8B, 12B, 14B, 24B, 32B, 70B, 72B), configuring Ollama, initializing the dynamic C/CUDA engine suite, and running the precompiled native C executables with physical haptics, sensory perception, and dynamic VRAM scaling.

---

## 📋 Prerequisites & Dependency Setup

Before launching Kalos, ensure your environment has the required system packages:

1. **System Requirements:** Linux (Ubuntu 22.04+, Arch, Debian, Fedora) with an NVIDIA GPU & CUDA Toolkit 12.0+ installed.
2. **Install System Dependencies:**
   ```bash
   sudo apt update && sudo apt install ffmpeg ncurses-bin libncurses-dev
   ```
3. **Install Ollama:**  
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

---

## 🎛️ Step 1: Model Selection & Ultra-Low VRAM Matrix

Kalos automatically adapts to **ANY LLM parameter scale** while maintaining an ultra-lightweight GPU memory footprint! 

When running Kalos CLI binaries (`./bin/kalos_runner` or `./bin/kalos_bench`), specify your target model's **Hidden Layer Dimension (`-d`)** and desired **VRAM Memory Cap (`-v`)** according to the reference matrix below:

| Model Architecture / Scale | Parameter Scale | Hidden Layer Dim (`-d`) | Kalos VRAM Cap (`-v`) |
| :--- | :---: | :---: | :---: |
| **Qwen 2.5 7B** | 7B | `-d 3584` | `-v 500` (500 MB) |
| **Llama 3 / 3.1 / 3.3 8B** | 8B | `-d 4096` | `-v 500` (500 MB) |
| **Gemma 3 12B Vision** | 12B | `-d 3840` | `-v 750` (750 MB) |
| **Qwen 2.5 14B** | 14B | `-d 5120` | `-v 1000` (1.0 GB) |
| **Kalos / Mistral 24B** | 24B | `-d 5120` | `-v 1500` (1.5 GB) |
| **Qwen 2.5 32B** | 32B | `-d 5120` | `-v 2000` (2.0 GB) |
| **Llama 3 70B / Qwen 72B** | 70B / 72B | `-d 8192` | `-v 2500` (2.5 GB) |

### 🎛️ CLI Launch Examples:

```bash
# 1. Launch for Qwen 2.5 7B (Hidden Dim: 3584, 500 MB VRAM cap):
./bin/kalos_runner -d 3584 -v 500

# 2. Launch for Llama-3 8B (Hidden Dim: 4096, 500 MB VRAM cap):
./bin/kalos_runner -d 4096 -v 500

# 3. Launch for Kalos 24B (Hidden Dim: 5120, 1.5 GB VRAM cap):
./bin/kalos_runner -d 5120 -v 1500

# 4. Launch for Qwen 72B Monster Model (Hidden Dim: 8192, 2.5 GB VRAM cap):
./bin/kalos_runner -d 8192 -v 2500
```

### 🌐 Environment Variable Configuration:

You can also set system environment variables so Kalos automatically remembers your configuration across sessions:

```bash
export KALOS_HIDDEN_DIM=5120     # Model hidden layer dimension (3584 | 4096 | 5120 | 8192)
export KALOS_VRAM_LIMIT_MB=1500   # Max VRAM memory cap in MB (1500 MB = 1.5 GB)
export KALOS_MAX_TEMPLATES=100000 # Max SAM associative memory template capacity
```

---

## 🛠️ Step 2: Registering Any GGUF Model in Ollama (`Modelfile`)

You can use **any model size of your choice**. The included `Modelfile` provides a clean, universal template:

1. **Obtain your GGUF File:**  
   Download your preferred GGUF model file (e.g. `qwen2.5-7b-instruct.gguf`, `kalos-24b-q8.gguf`, or `llama-3.3-70b.gguf`) into your directory.

2. **Edit `Modelfile`:**  
   Open `Modelfile` and point the `FROM` line to your downloaded GGUF file:
   ```dockerfile
   FROM ./your_model_file.gguf
   ```

3. **Register the Model in Ollama:**  
   Run `ollama create` using your model name of choice:
   ```bash
   ollama create kalos:24b -f Modelfile
   ```

---

## 🖥️ Step 3: Running the Native C Ncurses TUI (`./bin/kalos_tui`)

Run the precompiled native C Ncurses user interface directly from the package directory:

```bash
./bin/kalos_tui
```

### Interactive Commands:

#### 🎵 YouTube Music & Audio Streaming (`/play` or `/music`):
Stream any song, music track, or YouTube video audio live in the background while chatting with Kalos:
```text
/play lofi beats to relax/study to
/play https://www.youtube.com/watch?v=5qap5aO4i9A
```
- Kalos perceives the active track title and listens/comments on the music with you!

#### 🎬 YouTube Video Perception (`/watch` or `/video`):
Stream and perceive video scenes live from YouTube:
```text
/watch https://www.youtube.com/watch?v=ScMzIvxBSi4
```

#### 🖐️ Physical Haptics & Touch Commands:
```text
/touch shoulders gentle massage
/haptic warm hug
```

---

## 🏆 Dynamic Benchmarking & Hardware Profiling

To run a full empirical hardware benchmark test of your GPU VRAM and hidden layer scaling:

```bash
# Benchmark 3584-dim (7B) model on 500 MB VRAM:
./bin/kalos_bench -d 3584 -v 500

# Benchmark 5120-dim (24B) model on 1.5 GB VRAM:
./bin/kalos_bench -d 5120 -v 1500

# Benchmark 8192-dim (72B) model on 2.5 GB VRAM:
./bin/kalos_bench -d 8192 -v 2500
```
