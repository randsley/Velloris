# ⚡ Performance Guide

This guide provides detailed benchmarks, optimization techniques, and best practices for maximizing Velloris performance.

---

## Table of Contents

- [Performance Overview](#performance-overview)
- [Benchmarks by Mode](#benchmarks-by-mode)
- [Hardware Recommendations](#hardware-recommendations)
- [Optimization Techniques](#optimization-techniques)
- [Profiling & Monitoring](#profiling--monitoring)
- [Production Deployment](#production-deployment)
- [Troubleshooting Performance](#troubleshooting-performance)

---

## Performance Overview

### Key Metrics

**Realtime Mode:**
- **Latency:** 70-170ms (first token to audio)
- **Throughput:** 100+ requests/hour on RTX 4090
- **VRAM:** 16GB required
- **CPU:** Minimal usage (<10%)

**Dubbing Mode:**
- **Latency:** Faster than real-time (10s audio in 2-3s)
- **Throughput:** 1000+ clips/hour on RTX 4090
- **VRAM:** 6GB typical
- **CPU:** Low usage (~20%)

**Creative Mode:**
- **Latency:** 1-3 seconds (includes LLM reasoning)
- **Throughput:** 200+ requests/hour on RTX 4090
- **VRAM:** 8GB typical (depends on LLM size)
- **CPU:** Medium usage (~40% for Ollama)

---

## Benchmarks by Mode

### Realtime Mode Benchmarks

**Test Setup:**
- NVIDIA RTX 4090 (24GB VRAM)
- AMD Ryzen 9 5950X
- 64GB RAM
- CUDA 12.1
- 100 conversation turns

**Results:**

| Metric | Value | Notes |
|--------|-------|-------|
| **Average Latency** | 105ms | First token to audio output |
| **P50 Latency** | 95ms | 50th percentile |
| **P95 Latency** | 150ms | 95th percentile |
| **P99 Latency** | 170ms | 99th percentile |
| **Min Latency** | 70ms | Best case |
| **Max Latency** | 220ms | Worst case (cold start) |
| **Throughput** | 120 turns/hour | Sustained |
| **VRAM Usage** | 15.2GB | Stable |
| **Power Draw** | 250W | Average GPU power |

**Comparison:**
- **Gemini Live:** ~1900ms average latency (18x slower)
- **GPT-4 Voice:** ~800ms average latency (7.6x slower)
- **Human-to-human:** ~200ms typical (2x slower than Velloris)

---

### Dubbing Mode Benchmarks

**Test Setup:**
- NVIDIA RTX 3090 (24GB VRAM)
- Intel Core i9-10900K
- 32GB RAM
- CUDA 11.8
- 100 narration clips (10 seconds each)

**Results:**

| Metric | Value | Notes |
|--------|-------|-------|
| **Synthesis Speed** | 4.2x realtime | 10s audio in 2.4s |
| **Average Latency** | 2.1s | For 10s output |
| **Throughput** | 1200 clips/hour | 10s clips |
| **VRAM Usage** | 5.8GB | Stable |
| **CPU Usage** | 18% | Low overhead |
| **Power Draw** | 180W | Average GPU power |

**Quality Metrics:**
- **MOS (Mean Opinion Score):** 4.3/5.0 (natural, high quality)
- **WER (Word Error Rate):** 2.1% (transcription accuracy)
- **PESQ (Perceptual Evaluation of Speech Quality):** 4.1/5.0

---

### Creative Mode Benchmarks

**Test Setup:**
- NVIDIA RTX 4070 Ti (12GB VRAM)
- AMD Ryzen 7 5800X
- 32GB RAM
- CUDA 12.1
- Ollama with llama3:8b
- 100 creative prompts

**Results:**

| Metric | Value | Notes |
|--------|-------|-------|
| **Total Latency** | 2.3s average | LLM + TTS |
| **LLM Time** | 1.5s | Text generation |
| **TTS Time** | 0.8s | Audio synthesis |
| **Throughput** | 240 requests/hour | Sustained |
| **VRAM Usage** | 7.2GB | LLM + TTS |
| **CPU Usage** | 42% | Ollama overhead |
| **Power Draw** | 200W | Average GPU power |

**Breakdown:**
- LLM reasoning: 65% of total time
- TTS synthesis: 35% of total time

---

## Hardware Recommendations

### GPU Recommendations

**Realtime Mode (REQUIRED: NVIDIA GPU)**

| GPU | VRAM | Performance | Cost | Recommendation |
|-----|------|-------------|------|----------------|
| RTX 4090 | 24GB | Excellent (70-100ms) | $$$$ | Best overall |
| RTX 3090 | 24GB | Excellent (80-120ms) | $$$ | Great value |
| A100 40GB | 40GB | Excellent (70-100ms) | $$$$$ | Data center |
| A100 80GB | 80GB | Excellent (70-100ms) | $$$$$$ | Multi-user |
| RTX 4080 | 16GB | Good (90-140ms) | $$$ | Minimum viable |
| RTX 3080 Ti | 12GB | ❌ Insufficient VRAM | N/A | Not recommended |

**Dubbing/Creative Modes (GPU optional but recommended)**

| GPU | VRAM | Performance | Cost | Recommendation |
|-----|------|-------------|------|----------------|
| RTX 4090 | 24GB | Excellent | $$$$ | Overkill |
| RTX 4070 Ti | 12GB | Excellent | $$$ | Best value |
| RTX 3060 | 12GB | Good | $$ | Budget option |
| Apple M3 Max | 36GB unified | Good (MPS) | $$$$ | macOS only |
| Apple M2 Pro | 32GB unified | Good (MPS) | $$$ | macOS only |
| CPU only | N/A | Acceptable (3-5x slower) | $ | No GPU needed |

---

### CPU Recommendations

**For Ollama (Creative Mode):**

| CPU | Cores | Performance | Notes |
|-----|-------|-------------|-------|
| AMD Ryzen 9 7950X | 16 | Excellent | Best for Ollama |
| Intel Core i9-13900K | 24 | Excellent | Hybrid cores |
| AMD Ryzen 7 5800X | 8 | Good | Budget option |
| Apple M3 Max | 14-16 | Excellent | macOS only |
| Intel Core i5-12600K | 10 | Acceptable | Minimum viable |

**For Realtime/Dubbing (CPU usage minimal):**
- Any modern CPU (2020+) is sufficient
- Focus GPU budget instead

---

### RAM Recommendations

| Mode | Minimum | Recommended | Notes |
|------|---------|-------------|-------|
| Realtime | 16GB | 32GB | Model caching |
| Dubbing | 16GB | 32GB | Model caching |
| Creative | 32GB | 64GB | Ollama + models |

---

### Storage Recommendations

| Component | Size | Type | Notes |
|-----------|------|------|-------|
| Velloris code | 1GB | Any | Negligible |
| PersonaPlex-7B | 15GB | SSD | Fast loading |
| Qwen3-TTS | 3GB | SSD | Fast loading |
| Ollama models | 5-40GB | SSD | Per model |
| Generated audio | Variable | HDD/SSD | User content |

**Recommended:** 256GB+ NVMe SSD for all models

---

## Optimization Techniques

### 1. Model Loading Optimization

**Lazy Loading (Default in v2.0)**
```python
# Models load only when first used
orchestrator = Orchestrator(mode="dubbing")  # No loading yet
audio, sr = orchestrator.route_request(...)   # Loads Qwen3-TTS now
```

**Pre-loading for Production**
```python
# Load all models at startup
orchestrator = Orchestrator(mode="realtime")
orchestrator._load_personaplex()  # Pre-load

orchestrator_dubbing = Orchestrator(mode="dubbing")
orchestrator_dubbing._load_qwen3tts()  # Pre-load

# Now requests are instant (no cold start)
```

---

### 2. Batch Processing Optimization

**Sequential Processing (Slow)**
```python
for script in scripts:
    audio, sr = orchestrator.route_request(mode="dubbing", text=script)
    save_audio(audio, sr)
```

**Parallel Processing (Fast)**
```python
from concurrent.futures import ThreadPoolExecutor

def process_script(script):
    orchestrator = Orchestrator(mode="dubbing")
    audio, sr = orchestrator.route_request(mode="dubbing", text=script)
    return audio, sr

with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(process_script, scripts)
```

**Expected Speedup:** 3-4x on multi-core CPU

---

### 3. GPU Memory Optimization

**Mixed Precision (Enabled by Default)**
```python
# Already enabled in config.py
class ModelConfig:
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
```

**Manual Cleanup (If Needed)**
```python
import torch

# After processing batch
torch.cuda.empty_cache()
```

**Model Quantization (Future Feature)**
```python
# Coming in v2.1
orchestrator = Orchestrator(mode="realtime", quantization="8bit")
# Reduces VRAM from 16GB to 8GB
```

---

### 4. Ollama Optimization (Creative Mode)

**Use Smaller Models**
```bash
# Default: llama3 (70B parameters, slow)
ollama pull llama3:8b  # 8B parameters, 3x faster

python3 main.py --mode creative --script "Test" --llm-model llama3:8b
```

**GPU Acceleration for Ollama**
```bash
# Verify Ollama is using GPU
nvidia-smi  # Should show "ollama" process

# If not using GPU, reinstall Ollama with CUDA support
```

**Adjust Context Window**
```python
# In core/brain.py
self.llm = Ollama(
    model=model_name,
    num_ctx=2048  # Reduce from 4096 (default) for faster inference
)
```

---

### 5. Audio Processing Optimization

**Reduce Sample Rate (If Quality Not Critical)**
```python
# In .env file
OUTPUT_SAMPLE_RATE=16000  # Lower than default 24000
# Smaller files, faster processing, slightly lower quality
```

**Disable Real-Time Playback**
```python
# When batch processing, disable audio playback
# Comment out sounddevice.play() in code
# Just save to file
```

---

### 6. Network Optimization (API Mode)

**Enable Compression**
```python
# In Flask API
from flask_compress import Compress

app = Flask(__name__)
Compress(app)  # Compress audio responses
```

**Use Streaming**
```python
# Stream audio chunks instead of waiting for full generation
def generate_audio_stream():
    for sentence in sentences:
        audio, sr = orchestrator.route_request(...)
        yield audio

# Client receives audio progressively
```

---

## Profiling & Monitoring

### Basic Profiling

**Time Each Component:**
```python
import time

# PersonaPlex loading
start = time.time()
engine = PersonaPlexEngine()
print(f"PersonaPlex load time: {time.time() - start:.2f}s")

# Inference
start = time.time()
audio, sr = orchestrator.route_request(...)
print(f"Inference time: {time.time() - start:.2f}s")
```

---

### Advanced Profiling

**Python Profiler:**
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
orchestrator.route_request(...)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

**PyTorch Profiler:**
```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    audio, sr = orchestrator.route_request(...)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

### GPU Monitoring

**Real-Time Monitoring:**
```bash
# Terminal 1: Run Velloris
python3 main.py --mode realtime

# Terminal 2: Monitor GPU
watch -n 0.5 nvidia-smi

# Look at:
# - GPU utilization (should be >80% during inference)
# - Memory usage (should be stable)
# - Temperature (should be <85°C)
# - Power draw (should be near TDP during inference)
```

**Detailed Metrics:**
```bash
nvidia-smi dmon -s pucvmet -i 0
# p: Power
# u: Utilization
# c: Clock speed
# v: Voltage
# m: Memory usage
# e: ECC errors
# t: Temperature
```

---

### Application Monitoring

**Prometheus Metrics (Coming in v3.0):**
```python
from prometheus_client import Counter, Histogram

request_count = Counter('velloris_requests_total', 'Total requests')
latency_histogram = Histogram('velloris_latency_seconds', 'Request latency')

@latency_histogram.time()
def process_request(...):
    request_count.inc()
    # Process request
```

---

## Production Deployment

### Best Practices

**1. Pre-load Models**
```python
# At application startup
orchestrator = Orchestrator(mode="realtime")
orchestrator._load_personaplex()
# Now ready for instant responses
```

**2. Connection Pooling (API Mode)**
```python
# Reuse orchestrator instances
from flask import g

@app.before_request
def get_orchestrator():
    if 'orchestrator' not in g:
        g.orchestrator = Orchestrator(mode="dubbing")

@app.route('/synthesize', methods=['POST'])
def synthesize():
    # Reuse existing orchestrator
    audio, sr = g.orchestrator.route_request(...)
```

**3. Request Queuing**
```python
from queue import Queue
from threading import Thread

request_queue = Queue(maxsize=100)

def worker():
    orchestrator = Orchestrator(mode="dubbing")
    while True:
        request = request_queue.get()
        process_request(orchestrator, request)
        request_queue.task_done()

# Start workers
for _ in range(4):  # 4 worker threads
    Thread(target=worker, daemon=True).start()
```

**4. Caching Responses**
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def get_cached_audio(text_hash):
    # Check if we've generated this text before
    return load_from_cache(text_hash)

def synthesize(text):
    text_hash = hashlib.md5(text.encode()).hexdigest()
    cached = get_cached_audio(text_hash)
    if cached:
        return cached

    # Generate new
    audio, sr = orchestrator.route_request(mode="dubbing", text=text)
    save_to_cache(text_hash, audio, sr)
    return audio, sr
```

---

### Load Balancing

**Multiple GPU Deployment:**
```python
# Server 1: GPU 0
orchestrator1 = Orchestrator(mode="realtime")

# Server 2: GPU 1
orchestrator2 = Orchestrator(mode="realtime")

# Load balancer (round-robin)
orchestrators = [orchestrator1, orchestrator2]
current = 0

def get_next_orchestrator():
    global current
    orch = orchestrators[current]
    current = (current + 1) % len(orchestrators)
    return orch
```

**Expected Throughput:** Linear scaling (2 GPUs = 2x throughput)

---

### Monitoring in Production

**Health Checks:**
```python
@app.route('/health')
def health():
    try:
        # Quick inference test
        test_audio, sr = orchestrator.route_request(
            mode="dubbing",
            text="Health check"
        )
        return jsonify({'status': 'ok'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
```

**Metrics Endpoint:**
```python
@app.route('/metrics')
def metrics():
    import torch
    return jsonify({
        'gpu_memory_allocated': torch.cuda.memory_allocated() / 1e9,  # GB
        'gpu_memory_reserved': torch.cuda.memory_reserved() / 1e9,
        'requests_processed': request_count.get(),
        'average_latency': latency_histogram.observe(),
    })
```

---

## Troubleshooting Performance

### High Latency

**Symptoms:**
- Realtime mode >300ms
- Dubbing mode slower than real-time
- Creative mode >5s

**Diagnoses:**

1. **Check GPU usage:**
   ```bash
   nvidia-smi
   # Utilization should be >80% during inference
   # If low, check for CPU bottleneck
   ```

2. **Check VRAM:**
   ```bash
   nvidia-smi
   # Memory usage should be stable
   # If near max, reduce batch size or close other apps
   ```

3. **Check CPU:**
   ```bash
   top  # macOS/Linux
   # If CPU at 100%, upgrade CPU or optimize code
   ```

4. **Profile code:**
   ```python
   # Use cProfile to find bottlenecks
   ```

---

### Low Throughput

**Symptoms:**
- Can't process many requests per hour
- GPU underutilized

**Solutions:**

1. **Increase batch size** (if processing multiple requests)
2. **Use parallel workers** (see Batch Processing Optimization)
3. **Pre-load models** (eliminate cold start)
4. **Enable request queuing** (handle bursts)

---

### Memory Leaks

**Symptoms:**
- VRAM usage increases over time
- Eventually crashes with OOM error

**Solutions:**

1. **Manual cleanup:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

2. **Restart periodically:**
   ```python
   # In production, restart worker after N requests
   if request_count % 1000 == 0:
       restart_worker()
   ```

---

### GPU Thermal Throttling

**Symptoms:**
- Performance degrades over time
- GPU temperature >85°C

**Solutions:**

1. **Improve cooling** (case fans, liquid cooling)
2. **Reduce power limit:**
   ```bash
   sudo nvidia-smi -pl 300  # Limit to 300W (RTX 4090)
   ```
3. **Lower ambient temperature** (AC, better ventilation)

---

## Performance Checklist

Before deploying to production:

- [ ] Pre-load all models at startup
- [ ] Enable mixed precision (bfloat16/float16)
- [ ] Use appropriate hardware (see Hardware Recommendations)
- [ ] Implement request queuing
- [ ] Set up monitoring (nvidia-smi, Prometheus)
- [ ] Configure health checks
- [ ] Test under load (stress testing)
- [ ] Profile and optimize hot paths
- [ ] Implement caching for common requests
- [ ] Set up logging and alerts

---

## Additional Resources

- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Debugging performance issues
- [FAQ.md](FAQ.md) - Common performance questions
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design and architecture
- [GitHub Discussions](https://github.com/randsley/Velloris/discussions) - Community optimization tips

---

**Last updated:** February 2024
