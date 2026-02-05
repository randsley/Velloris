# Claude Code Guidelines for Velloris

## iOS/Xcode Conventions

When adding new Swift or Objective-C files to the project:
1. Create the source files in their appropriate directories
2. **Always update the Xcode `.pbxproj` file** to include the new file references and build file entries
3. Ensure the file is added to the correct target and build phase
4. After adding files, run `xcodebuild` to verify the project compiles without errors
5. Do not declare the task complete until the build succeeds

This prevents the common pitfall of creating source files that exist on disk but aren't referenced in the project file, causing build failures.

## Information Lookup

When asked to find, review, or reference project plans, documentation, specifications, or design notes:
1. **Always search markdown files first** using Glob patterns (`*.md`) and the Read tool
2. Check the following locations in order:
   - Project root (README.md, CLAUDE.md, design docs)
   - `/docs` directory (if it exists)
   - Any top-level `.md` files in the repository
3. Look for files containing keywords like "plan", "spec", "design", "roadmap", "architecture", or "todo"
4. Only use task list tools or other sources if markdown files don't contain the relevant information
5. Summarize what you found and cite the exact file and section

This ensures documentation is consulted from the actual source of truth in the repository.

## Testing & Validation

After making code changes:

**For Python scripts:**
- Always perform a quick syntax check: `python -c 'import ast; ast.parse(open("path/to/file.py").read())'`
- Or run `python -m py_compile path/to/file.py` to verify the file is syntactically valid
- For executable scripts, do a dry-run or `--help` check before declaring success
- Do not stop after fixing one error—trace through the entire execution path for cascading issues

**For Swift/Xcode projects:**
- After making changes, run `xcodebuild clean build` to verify compilation
- Check for any compiler warnings or errors
- If tests exist, run the test suite before declaring completion
- Verify that all modified files are included in the project file

Do not declare a fix complete until the code has been validated by running it or building it. Catching errors during this validation step prevents cascading failures in subsequent sessions.

## General Principles

- When executing Python scripts with issues, provide full error output upfront so all errors can be addressed in fewer iterations
- For multi-file changes, start with a checklist of all files that need to be created or modified, including project configuration files
- Iterate autonomously through fix cycles without stopping after each error—read the next error and continue fixing until the execution is clean

---

## Velloris Architecture Decisions (v2.0)

### Critical Architectural Rule: PersonaPlex Usage

**NEVER use PersonaPlex-7B for transcription only!**

PersonaPlex-7B is a full end-to-end Speech-to-Speech (S2S) model with built-in reasoning capabilities. Using it for transcription wastes 95% of its capabilities and defeats its purpose.

**❌ WRONG (v1.x mistake):**
```python
# BAD: Using PersonaPlex only for transcription
user_audio = capture_audio()
transcription = personaplex.transcribe_audio(user_audio)  # DEPRECATED!
llm_response = ollama.generate(transcription)
audio_response = qwen_tts.synthesize(llm_response)
```

**✅ CORRECT (v2.0 approach):**
```python
# GOOD: Using PersonaPlex for full end-to-end S2S
user_audio = capture_audio()
agent_audio, sr = personaplex.generate_s2s_response(
    audio=user_audio,
    voice_prompt="NATF2.pt",
    text_prompt="You are a helpful tutor"
)
# PersonaPlex handles: Listen → Understand → Reason → Respond → Speak (all in one!)
# Latency: 70-170ms (18x faster than cloud services)
```

### Three-Mode Architecture

Velloris v2.0 uses a **three-mode architecture** to properly utilize state-of-the-art models:

| Mode | When to Use | Models Used | Ollama Required? |
|------|-------------|-------------|------------------|
| **realtime** | Interactive conversations, customer service, low-latency chat | PersonaPlex-7B (end-to-end S2S) | ❌ No |
| **dubbing** | Content creation, video narration, audiobooks | Qwen3-TTS (high-fidelity synthesis) | ❌ No |
| **creative** | Emotional storytelling, creative content, LLM reasoning | Ollama + Qwen3-TTS | ✅ Yes |

### Mode Selection Guide for Contributors

When adding new features or fixing bugs, understand which mode is appropriate:

#### Use **realtime mode** when:
- You need ultra-low latency (70-170ms)
- Full-duplex conversation is required (natural interruptions)
- User needs to interrupt the agent mid-sentence
- You're building interactive voice applications
- Ollama dependency is not acceptable

#### Use **dubbing mode** when:
- You need professional-quality narration
- Multilingual support is required (10 languages)
- Voice cloning from reference audio is needed
- You're generating content for videos, podcasts, audiobooks
- LLM reasoning is NOT needed (just high-quality speech synthesis)

#### Use **creative mode** when:
- You need LLM reasoning and creativity (Ollama models: llama3, mistral, etc.)
- Emotion control via natural language is required
- You're building storytelling or creative applications
- Latency is less critical (1-3s acceptable)
- You can ensure Ollama is running

### Orchestrator Routing Pattern

All voice processing should go through `core/orchestrator.py`:

```python
from core.orchestrator import LocalVoiceOrchestrator

orchestrator = LocalVoiceOrchestrator()

# Real-time conversation
result = orchestrator.route_request(
    mode="realtime",
    audio_input=user_audio,
    voice_prompt="NATF2.pt",
    text_prompt="You are a helpful assistant"
)

# High-fidelity dubbing
result = orchestrator.route_request(
    mode="dubbing",
    text="Your narration script here",
    ref_audio_path="voices/reference.wav"  # Optional voice cloning
)

# Creative synthesis (requires Ollama)
result = orchestrator.route_request(
    mode="creative",
    text="Tell me a story about space",
    emotion="Speak with excitement"
)
```

### Lazy Loading Strategy

Models are loaded **only when first used** to minimize memory usage:

```python
orchestrator = LocalVoiceOrchestrator()  # No models loaded yet

# PersonaPlex loaded on first realtime mode call
orchestrator.route_request(mode="realtime", audio_input=audio)

# Qwen3-TTS loaded on first dubbing mode call
orchestrator.route_request(mode="dubbing", text="Hello")

# Unload models to free VRAM
orchestrator.unload_engines()
```

**Never pre-load models unless absolutely necessary.** This allows users to run Velloris on lower-end hardware by using only the modes they need.

### Error Handling: Ollama Dependency

Ollama is **OPTIONAL** in v2.0. Handle its absence gracefully:

```python
# ❌ BAD: Assume Ollama is available
from langchain_community.llms import Ollama
llm = Ollama(model="llama3")  # Will crash if Ollama not running!

# ✅ GOOD: Check availability first
try:
    from langchain_community.llms import Ollama
    llm = Ollama(model="llama3")
    print("ℹ️  Ollama available for creative mode")
except Exception as e:
    llm = None
    print("⚠️  Ollama not available. Creative mode disabled.")
    print("   Start with: ollama serve")
```

### Backward Compatibility

The old `interactive` mode is **deprecated but still supported** with warnings:

```python
if mode == "interactive":
    print("⚠️  WARNING: 'interactive' mode is deprecated!")
    print("   Use '--mode realtime' for PersonaPlex S2S (faster, full-duplex)")
    print("   Or '--mode creative' for Ollama + Qwen3-TTS (flexible, emotional)")
    print("   Defaulting to 'creative' mode...")
    mode = "creative"
```

When fixing bugs, **always encourage migration to the new modes** but don't break existing user scripts.

### Performance Expectations

Document performance characteristics in all PRs that affect latency:

- **realtime mode**: Target 70-170ms end-to-end latency
- **creative mode**: 1-3s latency (depends on LLM + TTS)
- **dubbing mode**: Non-interactive (no latency target)

### Testing Requirements

Before submitting PRs, ensure:

```bash
# 1. Run test suite (17 tests, all should pass in stub mode)
pytest tests/test_pipeline.py -v

# 2. Test each mode independently
python main.py --mode realtime --device cpu  # Test realtime
python main.py --mode dubbing --script "Test" --device cpu  # Test dubbing
# Note: creative mode requires Ollama running

# 3. Validate configuration
python main.py --show-config

# 4. Check for deprecation warnings
python main.py --mode interactive  # Should show deprecation warning
```

### Configuration Conventions

Always use `config.py` for centralized configuration:

```python
from config import Config

# ✅ GOOD: Use centralized config
voice = Config.app.REALTIME_VOICE
persona = Config.app.REALTIME_PERSONA
device = Config.model.DEVICE

# ❌ BAD: Hardcode values
voice = "NATF2"  # Don't hardcode!
device = "cuda"  # Don't hardcode!
```

### Documentation Standards

When adding features, update **ALL** relevant documentation:

1. **README.md** - User-facing usage examples
2. **ARCHITECTURE.md** - Technical implementation details
3. **MIGRATION.md** - If breaking changes, add migration instructions
4. **CLAUDE.md** - If architectural patterns change, document here

### Common Pitfalls to Avoid

1. **Using PersonaPlex for transcription only** → Use Whisper or the full S2S pipeline
2. **Requiring Ollama for all modes** → Only creative mode needs Ollama
3. **Hardcoding sample rates** → PersonaPlex=24kHz, Qwen3-TTS=12kHz (use Config)
4. **Not handling Ollama unavailability** → Always check and provide helpful error messages
5. **Breaking backward compatibility** → Deprecate with warnings, don't remove suddenly

### Reference Files

When working on Velloris, consult these files:

- **Architecture**: `ARCHITECTURE.md` - Full system design, mode comparison, performance metrics
- **Migration**: `MIGRATION.md` - v1.x to v2.0 migration guide
- **Refactor Plan**: `REFACTOR_PLAN.md` - Original refactor decisions and rationale
- **Configuration**: `config.py` - All configurable parameters
- **Orchestrator**: `core/orchestrator.py` - Mode-based routing logic
- **PersonaPlex**: `engines/personaplex.py` - End-to-end S2S implementation
- **Qwen3-TTS**: `engines/qwen_tts.py` - High-fidelity TTS implementation
