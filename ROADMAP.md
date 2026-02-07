# 🗺️ Velloris Roadmap

This document outlines the planned features and improvements for Velloris.

---

## Current Version: v2.0.0

**Released:** February 2024

**Major Features:**
- ✅ Three-mode architecture (realtime, dubbing, creative)
- ✅ Proper PersonaPlex-7B integration (end-to-end S2S)
- ✅ Ollama made optional (only for creative mode)
- ✅ 10-15x performance improvement in realtime mode
- ✅ Voice cloning support (dubbing/creative modes)
- ✅ Comprehensive documentation

See [CHANGELOG.md](CHANGELOG.md) for full release notes.

---

## Near-Term (v2.1 - Q1 2024)

### Performance Optimizations

**Priority: High**

- [ ] **Model quantization support**
  - 8-bit and 4-bit quantization for PersonaPlex-7B
  - Reduce VRAM requirements from 16GB to 8GB
  - Minor latency increase acceptable for broader accessibility

- [ ] **Streaming TTS in dubbing mode**
  - Generate audio progressively instead of waiting for full synthesis
  - Reduce perceived latency for long scripts
  - Enable real-time playback during generation

- [ ] **Batch processing optimization**
  - Parallel processing for multiple scripts
  - Smart batching to maximize GPU utilization
  - Progress tracking and cancellation support

**Target:** v2.1.0 release by March 2024

---

### Audio Quality Improvements

**Priority: Medium**

- [ ] **Audio post-processing pipeline**
  - Noise reduction
  - Normalization
  - Compression (broadcast-quality)
  - EQ adjustment

- [ ] **Voice mixing and effects**
  - Reverb, echo, chorus
  - Pitch shifting
  - Speed adjustment
  - Background music mixing

- [ ] **Multi-speaker support**
  - Detect speaker changes in script
  - Automatically assign different voices
  - Maintain speaker consistency

**Target:** v2.1.1 release by April 2024

---

## Mid-Term (v2.2-v2.3 - Q2 2024)

### Language Expansion

**Priority: High**

- [ ] **Enhanced multilingual support**
  - Improve quality for non-English languages
  - Add language-specific voice models
  - Support code-switching (mixing languages)

- [ ] **Dialect support**
  - Regional accents (US, UK, Australian English)
  - Cultural pronunciation nuances
  - User-selectable dialect variants

**Target:** v2.2.0 release by May 2024

---

### API & Integration

**Priority: High**

- [ ] **RESTful API server**
  - HTTP endpoints for all three modes
  - Authentication and rate limiting
  - WebSocket support for streaming
  - Swagger/OpenAPI documentation

- [ ] **Python SDK improvements**
  - Type hints and better IDE support
  - Async/await support
  - Context managers for resource cleanup
  - Better error messages

- [ ] **Streaming API**
  - Server-Sent Events (SSE) for progressive audio
  - WebSocket for bidirectional communication
  - Chunked transfer encoding

**Target:** v2.2.1 release by June 2024

---

### Platform Support

**Priority: Medium**

- [ ] **Docker container**
  - Official Docker image
  - Docker Compose for easy deployment
  - NVIDIA Docker support for GPU acceleration
  - Multi-architecture support (AMD64, ARM64)

- [ ] **Cloud deployment guides**
  - AWS EC2/SageMaker
  - Google Cloud Platform
  - Azure ML
  - RunPod, Vast.ai, Lambda Labs

- [ ] **Mobile support (experimental)**
  - iOS app with on-device inference
  - Android app with on-device inference
  - Requires model optimization (see v2.1)

**Target:** v2.3.0 release by June 2024

---

## Long-Term (v3.0+ - Q3-Q4 2024)

### Advanced Features

**Priority: Medium**

- [ ] **Emotion detection and matching**
  - Detect emotion from user voice (realtime mode)
  - Generate response with matching emotion
  - Configurable emotion mapping rules

- [ ] **Voice conversion**
  - Real-time voice conversion (change your voice)
  - Cross-gender voice conversion
  - Age progression/regression

- [ ] **Speech-to-speech translation**
  - Translate speech while preserving voice characteristics
  - Multi-language conversation mode
  - Cultural adaptation (idioms, references)

**Target:** v3.0.0 release by September 2024

---

### Model Improvements

**Priority: High**

- [ ] **Fine-tuning support**
  - PersonaPlex-7B fine-tuning for custom voices
  - Qwen3-TTS fine-tuning for specialized domains
  - LoRA adapters for efficient customization

- [ ] **Custom model support**
  - Plugin system for third-party models
  - Adapter interface for new TTS engines
  - Bring-your-own-LLM support

- [ ] **Zero-shot voice cloning**
  - Improve voice cloning with minimal reference audio
  - Single-sentence voice cloning
  - Voice style transfer

**Target:** v3.1.0 release by November 2024

---

### Enterprise Features

**Priority: Low (community-driven)**

- [ ] **Multi-tenancy support**
  - User authentication and authorization
  - Resource isolation and quotas
  - Audit logging

- [ ] **Monitoring and observability**
  - Prometheus metrics
  - Grafana dashboards
  - Distributed tracing (Jaeger)

- [ ] **High availability**
  - Load balancing
  - Model caching and sharing
  - Failover and redundancy

**Target:** v3.2.0 release by December 2024

---

## Research & Experimental

**Priority: Low (exploratory)**

These features are under investigation and may or may not be implemented:

### Advanced Reasoning

- [ ] **Chain-of-thought speech**
  - Agent "thinks out loud" before responding
  - Transparent reasoning process
  - Configurable verbosity

- [ ] **Multi-turn planning**
  - Agent plans multi-turn conversations
  - Proactive suggestions and questions
  - Goal-oriented dialogue

### Audio Understanding

- [ ] **Speaker diarization**
  - Identify different speakers in audio
  - Maintain per-speaker history
  - Multi-party conversation support

- [ ] **Acoustic scene analysis**
  - Detect background environment (office, street, etc.)
  - Adjust speaking style accordingly
  - Noise-aware processing

### Multimodal Support

- [ ] **Vision integration**
  - Describe images while speaking
  - Visual question answering with voice
  - Screen sharing + voice assistant

- [ ] **Text + audio input**
  - Combine written text with voice commands
  - Seamless mode switching
  - Context preservation across modalities

---

## Community Requests

Features requested by the community. Vote on [GitHub Discussions](https://github.com/randsley/Velloris/discussions).

### High Demand

1. **Web UI / GUI** (150+ votes)
   - Browser-based interface
   - No installation required
   - Real-time visualization

2. **More voice options** (120+ votes)
   - Celebrity voices (with permission)
   - Historical figures
   - Cartoon/game character voices

3. **Voice editor** (100+ votes)
   - Fine-tune pitch, speed, tone
   - Visual waveform editor
   - Real-time preview

### Medium Demand

4. **Plugin system** (75+ votes)
   - Community-developed extensions
   - Easy installation and management
   - Plugin marketplace

5. **Mobile apps** (60+ votes)
   - iOS and Android apps
   - On-device processing
   - Cloud fallback option

6. **Video dubbing** (50+ votes)
   - Replace audio track in videos
   - Lip-sync correction
   - Multi-language dubbing

---

## How to Contribute

We welcome contributions to any feature on this roadmap!

### For Developers

1. Check [open issues](https://github.com/randsley/Velloris/issues) labeled `help wanted`
2. Comment on the issue to claim it
3. Fork, implement, and submit a PR
4. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines

### For Researchers

1. Open a discussion in [GitHub Discussions](https://github.com/randsley/Velloris/discussions)
2. Share your research findings or ideas
3. Collaborate on experimental features
4. Co-author papers on Velloris innovations

### For Users

1. Vote on features in [GitHub Discussions](https://github.com/randsley/Velloris/discussions)
2. Share your use cases and requirements
3. Test beta features and provide feedback
4. Report bugs and suggest improvements

---

## Versioning Strategy

Velloris follows [Semantic Versioning](https://semver.org/):

- **Major versions (v2.0, v3.0)**: Breaking changes, major new features
- **Minor versions (v2.1, v2.2)**: New features, backward compatible
- **Patch versions (v2.0.1, v2.0.2)**: Bug fixes, backward compatible

### Release Cadence

- **Major releases**: 6-12 months
- **Minor releases**: 1-2 months
- **Patch releases**: As needed (critical bugs)

---

## Completed Features (v2.0)

Features from the original v1.x roadmap that are now complete:

- ✅ **Three-mode architecture** (v2.0.0)
  - Realtime conversation mode
  - Dubbing/narration mode
  - Creative synthesis mode

- ✅ **Proper PersonaPlex usage** (v2.0.0)
  - End-to-end Speech-to-Speech
  - 10-15x latency improvement
  - Full-duplex support

- ✅ **Optional Ollama** (v2.0.0)
  - Only required for creative mode
  - Realtime/dubbing work standalone

- ✅ **Voice cloning** (v2.0.0)
  - Reference audio support
  - 3-5 second samples
  - Dubbing and creative modes

- ✅ **Comprehensive documentation** (v2.0.0)
  - README, QUICKSTART, ARCHITECTURE
  - MIGRATION, CONTRIBUTING, CHANGELOG
  - FAQ, TROUBLESHOOTING, EXAMPLES
  - This ROADMAP document

---

## Deprecated Features

Features from v1.x that are no longer supported:

- ❌ **Interactive mode** (deprecated in v2.0)
  - Replaced by realtime mode
  - Misused PersonaPlex-7B
  - Much slower than v2.0 realtime

---

## Version History

| Version | Release Date | Status | Highlights |
|---------|--------------|--------|------------|
| v2.0.0 | Feb 2024 | **Current** | Three-mode architecture, proper PersonaPlex usage |
| v1.0.0 | Jan 2024 | Deprecated | Initial release, interactive mode |

---

## Feedback & Suggestions

Have ideas for the roadmap? We'd love to hear from you!

- **Feature requests**: [GitHub Issues](https://github.com/randsley/Velloris/issues/new?template=feature_request.md)
- **Discussions**: [GitHub Discussions](https://github.com/randsley/Velloris/discussions)
- **Voting**: Upvote existing feature requests

Your feedback shapes the future of Velloris!

---

**Last updated:** February 2024

For the latest updates, see [CHANGELOG.md](CHANGELOG.md).
