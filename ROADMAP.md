# 🗺️ Velloris Roadmap

This document outlines the planned features and improvements for Velloris.

---

## Current Version: v2.0.0

**Released:** February 2026

**Major Features:**
- ✅ Three-mode architecture (realtime, dubbing, creative)
- ✅ Proper PersonaPlex-7B integration (end-to-end S2S)
- ✅ Ollama made optional (only for creative mode)
- ✅ 10-15x performance improvement in realtime mode
- ✅ Voice cloning support (dubbing/creative modes)
- ✅ Comprehensive documentation

See [CHANGELOG.md](CHANGELOG.md) for full release notes.

---

## Near-Term (v2.1 - Q1 2026)

### Performance Optimizations

**Priority: High**

- [ ] **Model quantization support**
  - **Goal:** 8-bit and 4-bit quantization for PersonaPlex-7B.
  - **Outcome:** Reduce VRAM requirements from 16GB to 8GB with a latency increase of no more than 10%.

- [ ] **Streaming TTS in dubbing mode**
  - **Goal:** Generate audio progressively instead of waiting for full synthesis.
  - **Outcome:** Reduce perceived latency for long scripts by 50% and enable real-time playback during generation.

- [ ] **Batch processing optimization**
  - **Goal:** Parallel processing for multiple scripts.
  - **Outcome:** Smart batching to maximize GPU utilization, with progress tracking and cancellation support.

**Target:** v2.1.0 release by April 2026

---

### Audio Quality Improvements

**Priority: Medium**

- [ ] **Audio post-processing pipeline**
  - **Goal:** Implement a post-processing pipeline with noise reduction, normalization, compression, and EQ adjustment.
  - **Outcome:** Achieve broadcast-quality audio output.

- [ ] **Voice mixing and effects**
  - **Goal:** Add support for voice mixing and effects like reverb, echo, chorus, pitch shifting, speed adjustment, and background music mixing.
  - **Outcome:** Users can create more dynamic and engaging audio content.

- [ ] **Multi-speaker support**
  - **Goal:** Detect speaker changes in script and automatically assign different voices.
  - **Outcome:** Maintain speaker consistency in multi-speaker scripts.

**Target:** v2.1.1 release by May 2026

---

## Mid-Term (v2.2-v2.3 - Q2-Q3 2026)

### Language Expansion

**Priority: High**

- [ ] **Enhanced multilingual support**
  - **Goal:** Improve quality for non-English languages, with a focus on French, German, and Spanish.
  - **Outcome:** Achieve human-level quality for the target languages.
  - **Dependencies:** Language-specific voice models.

- [ ] **Dialect support**
  - **Goal:** Add support for regional accents and cultural pronunciation nuances.
  - **Outcome:** User-selectable dialect variants for English (US, UK, Australian), Spanish (Spain, Mexico), and French (France, Canada).

**Target:** v2.2.0 release by July 2026

---

### API & Integration

**Priority: High**

- [ ] **RESTful API server**
  - **Goal:** Implement a RESTful API server with endpoints for all three modes.
  - **Outcome:** Enable easy integration with other applications and services.

- [ ] **Python SDK improvements**
  - **Goal:** Improve the Python SDK with type hints, async/await support, context managers, and better error messages.
  - **Outcome:** A more developer-friendly and robust SDK.

- [ ] **Streaming API**
  - **Goal:** Implement a streaming API using Server-Sent Events (SSE) and WebSockets.
  - **Outcome:** Enable real-time, low-latency audio streaming.

**Target:** v2.2.1 release by August 2026

---

### Platform Support

**Priority: Medium**

- [ ] **Docker container**
  - **Goal:** Provide an official Docker image with multi-architecture support (AMD64, ARM64).
  - **Outcome:** Simplify deployment and ensure consistency across different environments.

- [ ] **Cloud deployment guides**
  - **Goal:** Create deployment guides for popular cloud platforms (AWS, GCP, Azure) and services (RunPod, Vast.ai, Lambda Labs).
  - **Outcome:** Lower the barrier to entry for cloud-based deployments.

- [ ] **Mobile support (experimental)**
  - **Goal:** Develop experimental iOS and Android apps with on-device inference.
  - **Outcome:** Showcase the potential of Velloris on mobile devices.
  - **Dependencies:** Model quantization support (v2.1).

**Target:** v2.3.0 release by September 2026

---

## Long-Term (v3.0+ - Q4 2026 and beyond)

### Advanced Features

**Priority: Medium**

- [ ] **Emotion detection and matching**
  - **Goal:** Detect emotion from user voice in realtime mode and generate a response with matching emotion.
  - **Outcome:** More natural and engaging conversations.

- [ ] **Voice conversion**
  - **Goal:** Implement real-time voice conversion, including cross-gender and age progression/regression.
  - **Outcome:** A powerful tool for content creators and privacy-conscious users.

- [ ] **Speech-to-speech translation**
  - **Goal:** Translate speech while preserving voice characteristics in a multi-language conversation mode.
  - **Outcome:** Break down language barriers in real-time communication.

**Target:** v3.0.0 release by Q4 2026

---

### Model Improvements

**Priority: High**

- [ ] **Fine-tuning support**
  - **Goal:** Add support for fine-tuning PersonaPlex-7B and Qwen3-TTS for custom voices and specialized domains.
  - **Outcome:** Users can create their own high-quality custom voices.

- [ ] **Custom model support**
  - **Goal:** Implement a plugin system for third-party models, including new TTS engines and LLMs.
  - **Outcome:** A more flexible and extensible platform.

- [ ] **Zero-shot voice cloning**
  - **Goal:** Improve voice cloning with minimal reference audio (e.g., a single sentence).
  - **Outcome:** Make voice cloning more accessible and easier to use.

**Target:** v3.1.0 release by Q1 2027

---

### Enterprise Features

**Priority: Low (community-driven)**

- [ ] **Multi-tenancy support**
- [ ] **Monitoring and observability**
- [ ] **High availability**

**Target:** v3.2.0 release by Q2 2027

---

## Research & Experimental

**Priority: Low (exploratory)**

These features are under investigation and may or may not be implemented:

### Advanced Reasoning
- [ ] **Chain-of-thought speech**
- [ ] **Multi-turn planning**

### Audio Understanding
- [ ] **Speaker diarization**
- [ ] **Acoustic scene analysis**

### Multimodal Support
- [ ] **Vision integration**
- [ ] **Text + audio input**

---

## Community Requests

Features requested by the community. Vote on [GitHub Discussions](https://github.com/randsley/Velloris/discussions). The top 3 most upvoted features will be considered for the next release cycle.

### High Demand
1. **Web UI / GUI** (150+ votes)
2. **More voice options** (120+ votes)
3. **Voice editor** (100+ votes)

### Medium Demand
4. **Plugin system** (75+ votes)
5. **Mobile apps** (60+ votes)
6. **Video dubbing** (50+ votes)

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
- **Minor releases**: 1-3 months
- **Patch releases**: As needed (critical bugs)

---

## Completed Features (v2.0)

Features from the original v1.x roadmap that are now complete:

- ✅ **Three-mode architecture** (v2.0.0)
- ✅ **Proper PersonaPlex usage** (v2.0.0)
- ✅ **Optional Ollama** (v2.0.0)
- ✅ **Voice cloning** (v2.0.0)
- ✅ **Comprehensive documentation** (v2.0.0)

---

## Deprecated Features

Features from v1.x that are no longer supported:

- ❌ **Interactive mode** (deprecated in v2.0)

---

## Version History

| Version | Release Date | Status | Highlights |
|---------|--------------|--------|------------|
| v2.0.0 | Feb 2026 | **Current** | Three-mode architecture, proper PersonaPlex usage |
| v1.0.0 | Jan 2026 | Deprecated | Initial release, interactive mode |

---

## Feedback & Suggestions

Have ideas for the roadmap? We'd love to hear from you!

- **Feature requests**: [GitHub Issues](https://github.com/randsley/Velloris/issues/new?template=feature_request.md)
- **Discussions**: [GitHub Discussions](https://github.com/randsley/Velloris/discussions)
- **Voting**: Upvote existing feature requests

Your feedback shapes the future of Velloris!

---

**Last updated:** February 2026

For the latest updates, see [CHANGELOG.md](CHANGELOG.md).
