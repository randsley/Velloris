# Contributing to Velloris

Thank you for your interest in contributing to Velloris! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Architectural Guidelines](#architectural-guidelines)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful, constructive, and professional in all interactions.

---

## Getting Started

### Prerequisites

- **Python 3.12+** (3.11+ supported)
- **Git** for version control
- **For Real-Time Mode**: NVIDIA GPU (16GB+ VRAM) + CUDA 12.1+
- **For Creative Mode**: Ollama installed and running
- **macOS**: Homebrew for system dependencies

### First Steps

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Velloris.git
   cd Velloris
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/randsley/Velloris.git
   ```

---

## Development Setup

### macOS

```bash
chmod +x install_macos.sh
./install_macos.sh

# Activate virtual environment
source venv_py312/bin/activate
```

### Windows

```bash
install_windows.bat

# Activate virtual environment
venv_py312\Scripts\activate
```

### Verify Installation

```bash
python main.py --show-config
pytest tests/test_pipeline.py -v
```

---

## How to Contribute

### Reporting Bugs

1. **Check existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (`python main.py --show-config`)
   - Error messages and stack traces

### Suggesting Features

1. **Check ROADMAP.md** for planned features
2. **Create a feature request issue** with:
   - Clear description of the feature
   - Use case and motivation
   - Proposed implementation (if applicable)
   - Which mode it affects (realtime/dubbing/creative)

### Submitting Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

2. **Make your changes** following coding standards

3. **Test your changes**:
   ```bash
   pytest tests/test_pipeline.py -v
   python main.py --mode dubbing --script "Test" --device cpu
   ```

4. **Commit your changes**:
   ```bash
   git add <files>
   git commit -m "Brief description of changes"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** on GitHub

---

## Coding Standards

### Python Style Guide

- Follow **PEP 8** style guidelines
- Use **type hints** for function parameters and returns
- Maximum line length: **100 characters**
- Use **descriptive variable names**

### Code Organization

```python
# Standard library imports
import os
from pathlib import Path

# Third-party imports
import numpy as np
import torch

# Local imports
from core.orchestrator import LocalVoiceOrchestrator
from config import Config
```

### Documentation

- **Docstrings** for all classes and public methods (Google style):
  ```python
  def generate_s2s_response(
      self,
      audio: np.ndarray,
      sr: int = 24000,
      voice_prompt: Optional[str] = None
  ) -> Optional[Tuple[np.ndarray, int]]:
      """
      Generate end-to-end speech-to-speech response.

      Args:
          audio: User audio input (24kHz preferred)
          sr: Input sample rate
          voice_prompt: Voice embedding file (e.g., "NATF2.pt")

      Returns:
          Tuple of (agent_audio, sample_rate) or None
      """
  ```

- **Comments** for complex logic
- **Update documentation** when changing behavior

### Configuration

- **Never hardcode values** - use `config.py`:
  ```python
  # ✅ GOOD
  from config import Config
  voice = Config.app.REALTIME_VOICE

  # ❌ BAD
  voice = "NATF2"
  ```

- **Add new config options** to `config.py` and `.env.example`

---

## Testing Requirements

### Before Submitting a PR

```bash
# 1. Run test suite (all tests must pass)
pytest tests/test_pipeline.py -v

# 2. Test each affected mode
python main.py --mode realtime --device cpu
python main.py --mode dubbing --script "Test narration" --device cpu
python main.py --mode creative --script "Test" --device cpu  # Requires Ollama

# 3. Validate configuration
python main.py --show-config

# 4. Check for deprecation warnings
python main.py --mode interactive  # Should show deprecation warning
```

### Writing Tests

- Add tests for new features in `tests/`
- Ensure tests pass in **stub mode** (without models)
- Test both success and error cases
- Use descriptive test names:
  ```python
  def test_orchestrator_routes_realtime_mode_correctly():
      """Test that orchestrator correctly routes to realtime mode."""
  ```

---

## Pull Request Process

### PR Title Format

Use conventional commit format:

- `feat: Add voice cloning support for realtime mode`
- `fix: Resolve Ollama connection timeout in creative mode`
- `docs: Update ARCHITECTURE.md with performance benchmarks`
- `refactor: Simplify orchestrator routing logic`
- `test: Add tests for PersonaPlex S2S generation`
- `chore: Update dependencies in requirements.txt`

### PR Description Template

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Affected Modes
- [ ] Realtime mode
- [ ] Dubbing mode
- [ ] Creative mode
- [ ] All modes

## Testing Done
- [ ] Ran test suite (`pytest tests/test_pipeline.py -v`)
- [ ] Tested affected mode(s) manually
- [ ] Validated configuration (`python main.py --show-config`)
- [ ] Checked for deprecation warnings

## Documentation Updated
- [ ] README.md
- [ ] ARCHITECTURE.md
- [ ] MIGRATION.md (if breaking change)
- [ ] CHANGELOG.md
- [ ] Code comments/docstrings

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Documentation updated
```

### Review Process

1. **Automated checks** must pass (if CI/CD is configured)
2. **Code review** by maintainer(s)
3. **Address feedback** and push updates
4. **Approval** and merge by maintainer

---

## Architectural Guidelines

### Critical Rules

⚠️ **NEVER use PersonaPlex-7B for transcription only!**

PersonaPlex is a full end-to-end Speech-to-Speech model. Using it only for transcription wastes 95% of its capabilities.

```python
# ❌ WRONG
transcription = personaplex.transcribe_audio(user_audio)  # DEPRECATED!
llm_response = ollama.generate(transcription)

# ✅ CORRECT
agent_audio, sr = personaplex.generate_s2s_response(user_audio)
```

### Three-Mode Architecture

| Mode | When to Use | Models Used | Ollama Required? |
|------|-------------|-------------|------------------|
| **realtime** | Interactive conversations | PersonaPlex-7B S2S | ❌ No |
| **dubbing** | Content creation, narration | Qwen3-TTS | ❌ No |
| **creative** | Emotional storytelling | Ollama + Qwen3-TTS | ✅ Yes |

### Mode Selection Guide

- **Realtime mode**: Ultra-low latency (70-170ms), full-duplex, interruptions
- **Dubbing mode**: Professional quality, multilingual, voice cloning
- **Creative mode**: LLM reasoning, emotion control, creative content

### Orchestrator Pattern

All voice processing should go through `core/orchestrator.py`:

```python
from core.orchestrator import LocalVoiceOrchestrator

orchestrator = LocalVoiceOrchestrator()

# Real-time conversation
result = orchestrator.route_request(
    mode="realtime",
    audio_input=user_audio,
    voice_prompt="NATF2.pt"
)
```

### Lazy Loading

Models should be loaded **only when first used**:

```python
# ✅ GOOD: Load on demand
orchestrator = LocalVoiceOrchestrator()  # No models loaded
orchestrator.route_request(mode="realtime", audio_input=audio)  # PersonaPlex loads now

# ❌ BAD: Pre-load models
orchestrator = LocalVoiceOrchestrator()
orchestrator._load_personaplex()  # Wastes memory if not used
```

### Error Handling

Ollama is **OPTIONAL** in v2.0. Handle its absence gracefully:

```python
# ✅ GOOD
try:
    from langchain_community.llms import Ollama
    llm = Ollama(model="llama3")
except Exception as e:
    llm = None
    print("⚠️  Ollama not available. Creative mode disabled.")
```

### Backward Compatibility

- **Deprecate** old features with warnings
- **Don't break** existing user scripts
- **Provide migration path** in MIGRATION.md

### Performance Expectations

Document performance characteristics:

- **realtime mode**: Target 70-170ms end-to-end latency
- **creative mode**: 1-3s latency (depends on LLM + TTS)
- **dubbing mode**: Non-interactive (no latency target)

### Documentation Standards

When adding features, update **ALL** relevant documentation:

1. **README.md** - User-facing usage examples
2. **ARCHITECTURE.md** - Technical implementation details
3. **MIGRATION.md** - If breaking changes, add migration instructions
4. **CHANGELOG.md** - Version tracking
5. **This file** - If contribution process changes

---

## Common Pitfalls to Avoid

1. ❌ **Using PersonaPlex for transcription only** → Use full S2S pipeline
2. ❌ **Requiring Ollama for all modes** → Only creative mode needs Ollama
3. ❌ **Hardcoding sample rates** → Use Config (PersonaPlex=24kHz, Qwen3=12kHz)
4. ❌ **Not handling Ollama unavailability** → Always check and provide helpful errors
5. ❌ **Breaking backward compatibility** → Deprecate with warnings

---

## Reference Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design, mode comparison, performance
- **[MIGRATION.md](MIGRATION.md)** - v1.x to v2.0 migration guide
- **[REFACTOR_PLAN.md](REFACTOR_PLAN.md)** - Original refactor decisions
- **[CLAUDE.md](CLAUDE.md)** - Additional architectural guidance
- **[FAQ.md](FAQ.md)** - Common questions and answers
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Debugging guide

---

## Questions?

- **Read the docs**: Start with [README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md)
- **Search issues**: Check [GitHub Issues](https://github.com/randsley/Velloris/issues)
- **Ask questions**: Open a Discussion on GitHub
- **Join community**: [Link to Discord/Slack if available]

---

## License

By contributing to Velloris, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

---

**Thank you for contributing to Velloris! 🎙️**
