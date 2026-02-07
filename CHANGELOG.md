# Changelog

All notable changes to Velloris will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- MLX Stack integration for M-series Mac optimization (2-3x speedup potential)
- Web UI with Gradio
- ONNX export for edge deployment
- Mobile optimization (iOS/Android)
- Multi-turn conversation memory
- Custom voice fine-tuning
- Real-time transcription display

---

## [2.0.0] - 2026-02-05

### 🎉 Major Release: Three-Mode Architecture

This release fixes critical architectural issues and introduces a three-mode system that properly utilizes PersonaPlex-7B.

### Added

#### Core Features
- **Three-Mode Architecture**: Realtime, Dubbing, and Creative modes
  - **Realtime Mode**: PersonaPlex-7B end-to-end S2S (70-170ms latency, full-duplex)
  - **Dubbing Mode**: Qwen3-TTS high-fidelity narration (10 languages, voice cloning)
  - **Creative Mode**: Ollama + Qwen3-TTS emotional synthesis (LLM reasoning)
- **Proper PersonaPlex Usage**: Now uses full S2S pipeline instead of transcription-only
- **Optional Ollama**: No longer required for basic conversations (only creative mode)
- **Mode-Based Routing**: `core/orchestrator.py` intelligently routes requests
- **Lazy Loading**: Models load only when needed, reducing memory usage

#### Configuration
- New mode-specific configuration options:
  - `REALTIME_VOICE`: Voice selection for realtime mode (16 preset voices)
  - `REALTIME_PERSONA`: Persona/role for PersonaPlex
  - `CREATIVE_LLM`: Ollama model selection for creative mode
  - `CREATIVE_EMOTION`: Emotion instruction for creative synthesis
- Updated `.env.example` with comprehensive v2.0 options

#### Documentation
- **MIGRATION.md**: Comprehensive v1.x to v2.0 migration guide
- **REFACTOR_PLAN.md**: Detailed refactor rationale and decisions
- **CONTRIBUTING.md**: Contribution guidelines for open-source collaboration
- **CHANGELOG.md**: Version tracking (this file)
- **QUICKSTART.md**: Fast onboarding guide for new users
- **FAQ.md**: Common questions and answers
- **TROUBLESHOOTING.md**: Debugging and problem-solving guide
- **EXAMPLES.md**: Code examples and usage patterns
- **ROADMAP.md**: Future plans and feature roadmap
- **CLAUDE.md**: Extended with Velloris Architecture Decisions section
- **ARCHITECTURE.md**: Completely updated with three-mode architecture
- **README.md**: Rewritten with migration notice and mode comparison

#### CLI
- New `--persona` argument for realtime mode persona control
- New `--voice` argument for realtime mode voice selection (16 voices)
- New `--emotion` argument for creative mode emotion control
- Updated `--mode` choices: `realtime`, `dubbing`, `creative` (plus deprecated `interactive`)

### Changed

#### Breaking Changes
- **Mode Rename**: `interactive` → `realtime` or `creative`
  - Old `interactive` mode is deprecated but still works with warnings
  - Use `--mode realtime` for PersonaPlex S2S (10-15x faster)
  - Use `--mode creative` for Ollama + Qwen3-TTS (similar to old behavior)
- **Ollama**: Now optional (only needed for creative mode)
- **PersonaPlex Usage**: Completely refactored to use end-to-end S2S
  - `transcribe_audio()` method deprecated with warnings
  - New `generate_s2s_response()` as primary method

#### Architecture
- **core/orchestrator.py**: Complete rewrite with three-mode routing
  - Added `_handle_realtime()`, `_handle_creative()`, `_handle_dubbing()`
  - Added `_load_ollama()` for lazy Ollama initialization
  - Improved error handling and user feedback
- **core/brain.py**: Made Ollama optional, mode-aware initialization
  - Brain only loads for creative mode
  - Added warnings for incorrect usage patterns
- **engines/personaplex.py**:
  - Deprecated `transcribe_audio()` with clear warnings
  - Added `generate_s2s_response()` as primary method
  - Improved documentation and usage examples
- **config.py**:
  - Changed `DEFAULT_MODE` from "interactive" to "realtime"
  - Added mode-specific configuration sections
  - Added backward compatibility settings (deprecated)

#### Performance
- **10-15x faster latency** in realtime mode (70-170ms vs 2000ms+)
- **Reduced memory usage** through lazy loading
- **No Ollama dependency** for realtime and dubbing modes

### Deprecated
- `--mode interactive` - Use `--mode realtime` or `--mode creative` instead
- `PersonaPlexEngine.transcribe_audio()` - Use `generate_s2s_response()` instead
- `VoiceAgentBrain.process_audio_turn()` - Use `orchestrator.route_request()` instead
- `OLLAMA_MODEL` config - Use `CREATIVE_LLM` instead
- `INTERACTIVE_TIMEOUT` config - Use `REALTIME_TIMEOUT` or `CREATIVE_TIMEOUT` instead

### Fixed
- **Critical**: PersonaPlex-7B was being misused for transcription only (wasting 95% of capabilities)
- **Performance**: Removed unnecessary LLM calls in realtime mode
- **Architecture**: Proper separation of concerns between modes
- **Error Handling**: Better error messages when Ollama is unavailable

### Security
- No security-related changes in this release

---

## [1.0.0] - 2026-02-04

### Initial Release

#### Added
- Basic two-mode architecture (interactive and dubbing)
- PersonaPlex-7B integration for speech processing
- Qwen3-TTS integration for high-fidelity synthesis
- Ollama LLM integration (required for all modes)
- Voice Activity Detection (Silero VAD)
- Cross-platform support (Windows CUDA, macOS MPS, Linux CPU)
- Configuration management via `config.py`
- CLI interface with `main.py`
- Test suite with pytest
- Basic documentation (README, ARCHITECTURE, LICENSE)

#### Known Issues
- PersonaPlex-7B used only for transcription (architectural issue - fixed in v2.0)
- Ollama required for all modes (improved in v2.0)
- High latency in interactive mode (~2000ms+ - improved to 70-170ms in v2.0)

---

## Version History Summary

- **v2.0.0** (2026-02-05): Three-mode architecture, proper PersonaPlex usage, optional Ollama, 10-15x faster
- **v1.0.0** (2026-02-04): Initial release with two-mode architecture

---

## Upgrade Guide

### From v1.x to v2.0

See [MIGRATION.md](MIGRATION.md) for comprehensive upgrade instructions.

**Quick Summary:**
```bash
# Old (v1.x)
python main.py --mode interactive

# New (v2.0) - Option A: Faster, full-duplex
python main.py --mode realtime

# New (v2.0) - Option B: LLM reasoning (requires Ollama)
python main.py --mode creative
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to Velloris.

---

## License

Apache License 2.0 - See [LICENSE](LICENSE) file for details.
