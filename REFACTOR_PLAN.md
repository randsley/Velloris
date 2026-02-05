# Velloris Architecture Refactor Plan

**Date:** 2026-02-05
**Status:** Planning Phase
**Priority:** High - Fixes fundamental architectural misuse

---

## 🚨 Critical Issue Identified

### Current Problem: PersonaPlex is Severely Underutilized

**What's Happening:**
- PersonaPlex-7B is being used **only for transcription** (line 139 in `core/brain.py`)
- This is like using a Ferrari to deliver pizza 🍕🏎️

**The Issue:**
```python
# core/brain.py:139
user_text = self.orchestrator.personaplex_engine.transcribe_audio(audio, sr)
```

**What PersonaPlex Actually Is:**
- ✅ Full end-to-end Speech-to-Speech (S2S) model
- ✅ Listens, understands, reasons, and responds
- ✅ Generates intelligent audio responses directly
- ✅ 70-170ms ultra-low latency
- ✅ True full-duplex (interruptions, backchanneling)
- ✅ 7B parameter conversational AI

**Current Pipeline (WRONG):**
```
User Speech → PersonaPlex (transcription only)
    → Ollama LLM (reasoning)
    → Qwen3-TTS (synthesis)
    → Audio Output

Latency: ~2000ms+ (cascade delay)
Full-Duplex: ❌ No
PersonaPlex Capabilities Used: ~5%
```

**What It Should Be:**
```
User Speech → PersonaPlex (end-to-end S2S) → Audio Output

Latency: 70-170ms ⚡
Full-Duplex: ✅ Yes
PersonaPlex Capabilities Used: 100%
```

---

## 📊 Analysis of Current Implementation

### File: `engines/personaplex.py`

**Correct Parts:**
- ✅ Lines 1-22: Excellent documentation
- ✅ Lines 58-78: All 16 voices properly defined
- ✅ Lines 169-209: `process_speech()` method exists (but not used)
- ✅ Lines 288-325: `process_voice_turn()` method exists (but not used)

**Incorrect Parts:**
- ❌ Lines 145-167: `transcribe_audio()` - This method shouldn't be primary use
- ❌ Lines 211-248: `generate_speech()` - Delegates to Qwen3-TTS (confusing)
- ❌ Line 165: Comment admits "PersonaPlex is optimized for speech-to-speech"
- ❌ Implementation: Stub mode returns empty strings/silence

**Key Quote from Code (line 165-166):**
```python
print("[INFO] PersonaPlex-7B is optimized for speech-to-speech.")
print("       For text transcription, use Whisper or another STT model.")
```
**This comment proves the developer knew PersonaPlex was being misused!**

### File: `core/brain.py`

**Line 138-139 (THE PROBLEM):**
```python
# Step 1: Transcribe audio using PersonaPlex
user_text = self.orchestrator.personaplex_engine.transcribe_audio(audio, sr)
```

This line:
1. ❌ Uses PersonaPlex for transcription only
2. ❌ Ignores PersonaPlex's response generation capability
3. ❌ Forces unnecessary Ollama dependency
4. ❌ Adds 1500ms+ latency from cascade
5. ❌ Prevents full-duplex conversations

**What Should Happen:**
```python
# Step 1: PersonaPlex handles EVERYTHING (listen + understand + respond)
response_audio, sr = self.orchestrator.personaplex_engine.process_voice_turn(audio, sr)
```

---

## 🎯 Proposed Solution: Three-Mode Architecture

### Mode 1: **Real-Time Conversation** (PersonaPlex Native)
```
User Speech → PersonaPlex-7B (end-to-end S2S) → Audio Output
```

**Best For:**
- Interactive voice assistants
- Customer service bots
- Live tutoring/education
- Gaming NPCs with voice
- Emergency response systems

**Benefits:**
- ⚡ 70-170ms latency (18x faster than Gemini Live)
- ✅ True full-duplex (95% interruption success)
- ✅ Natural backchanneling
- ✅ No Ollama dependency
- ✅ Simpler architecture

**Requirements:**
- NVIDIA GPU (16GB+ VRAM)
- CUDA support
- English language only

**CLI:**
```bash
python main.py --mode realtime --persona "You are a helpful tutor" --voice NATF2
```

---

### Mode 2: **High-Fidelity Dubbing** (Qwen3-TTS) ✅ No Changes
```
Script Text → Qwen3-TTS → Audio Output
```

**Best For:**
- Audiobook narration
- Video dubbing
- Podcast creation
- Content creation
- Multilingual voiceovers

**Benefits:**
- 🎨 Professional quality (12kHz)
- 🌍 10 languages supported
- 🎭 Emotion control via instructions
- 🗣️ Voice cloning (3-second samples)
- 🎨 Voice design via natural language
- 💻 Works on CUDA/MPS/CPU

**Requirements:**
- GPU recommended (6-12GB VRAM)
- CPU fallback available

**CLI:**
```bash
python main.py --mode dubbing --script "Your narration here"
```

---

### Mode 3: **Creative Assistant** (Ollama + Qwen3-TTS) 🆕 NEW
```
User Text → Ollama LLM (reasoning) → Qwen3-TTS (emotional synthesis) → Audio Output
```

**Best For:**
- Creative writing/storytelling
- Emotional content generation
- Brainstorming sessions
- Character voice acting
- Multilingual conversations with LLM reasoning

**Benefits:**
- 🧠 Full LLM reasoning (Ollama flexibility)
- 🎭 Emotion control via Qwen3-TTS instructions
- 🌍 Multilingual (10 languages)
- 🎨 Voice design capabilities
- 🔒 Local-first (privacy maintained)

**Requirements:**
- Ollama running (`ollama serve`)
- GPU recommended for Qwen3-TTS
- Internet for initial model downloads

**CLI:**
```bash
# Terminal 1
ollama serve

# Terminal 2
python main.py --mode creative --emotion "Speak with excitement"
```

---

## 📋 Detailed Task Breakdown

### Phase 1: Core Architecture Refactor (CRITICAL)

#### Task 1.1: Fix `engines/personaplex.py`
**File:** `engines/personaplex.py`
**Lines to modify:** 145-167, 169-209, 288-325

**Changes:**
1. Deprecate `transcribe_audio()` method
2. Make `process_voice_turn()` the primary method
3. Implement full S2S logic (remove stub)
4. Add proper error handling
5. Add streaming support

**New Primary API:**
```python
def process_voice_turn(
    self,
    audio: np.ndarray,
    sr: int = 24000,
    voice_prompt: str = "NATF2.pt",
    text_prompt: str = "You are a helpful assistant.",
    streaming: bool = True
) -> Tuple[np.ndarray, int]:
    """
    End-to-end speech-to-speech processing with PersonaPlex-7B.

    This is the PRIMARY method for PersonaPlex usage.
    No separate LLM or TTS needed.

    Args:
        audio: User speech (24kHz)
        sr: Sample rate
        voice_prompt: Voice file (NATF0-3, NATM0-3, VARF0-4, VARM0-4)
        text_prompt: Persona/role description
        streaming: Enable real-time streaming

    Returns:
        Tuple of (agent_audio, sample_rate)
    """
```

#### Task 1.2: Refactor `core/orchestrator.py`
**File:** `core/orchestrator.py`
**New method:** `route_request(mode, **kwargs)`

**Changes:**
1. Add mode-based routing logic
2. Create `_handle_realtime()` for PersonaPlex S2S
3. Keep `_handle_dubbing()` for Qwen3-TTS (unchanged)
4. Add `_handle_creative()` for Ollama + Qwen3-TTS
5. Update lazy loading per mode

**New API:**
```python
def route_request(self, mode: str, **kwargs):
    """
    Route to appropriate engine based on mode.

    Modes:
    - 'realtime': PersonaPlex full-duplex S2S (no LLM)
    - 'dubbing': Qwen3-TTS high-fidelity narration
    - 'creative': Ollama + Qwen3-TTS for emotional content
    """
    if mode == "realtime":
        return self._handle_realtime(kwargs)
    elif mode == "dubbing":
        return self._handle_dubbing(kwargs)
    elif mode == "creative":
        return self._handle_creative(kwargs)
```

#### Task 1.3: Update `core/brain.py`
**File:** `core/brain.py`
**Lines to modify:** 28-40, 120-148

**Changes:**
1. Make Ollama optional (only for creative mode)
2. Add mode parameter to `__init__()`
3. Update `process_audio_turn()` to use PersonaPlex correctly
4. Remove line 139 (transcribe_audio call)

**New Constructor:**
```python
def __init__(
    self,
    mode: str = "realtime",
    model_name: str = "llama3",
    tts_engine=None,
    orchestrator=None
):
    """
    Initialize the brain with mode-based architecture.

    Args:
        mode: Operating mode ('realtime', 'dubbing', 'creative')
        model_name: Ollama model (only used in creative mode)
        tts_engine: Optional TTS engine
        orchestrator: Required for realtime mode
    """
    self.mode = mode

    # Ollama only needed for creative mode
    if mode == "creative":
        self.llm = Ollama(model=model_name)
    else:
        self.llm = None
        if mode == "realtime":
            print("ℹ️  Running in real-time mode (PersonaPlex handles reasoning)")
```

#### Task 1.4: Update `main.py` CLI
**File:** `main.py`
**Lines to modify:** CLI argument parser

**Changes:**
1. Change `--mode` choices from `["interactive", "dubbing"]` to `["realtime", "dubbing", "creative"]`
2. Add `--persona` argument for PersonaPlex
3. Add `--voice` argument for PersonaPlex
4. Add `--emotion` argument for creative mode
5. Update help text to explain modes

**New CLI:**
```python
parser.add_argument(
    "--mode",
    type=str,
    choices=["realtime", "dubbing", "creative"],
    default="realtime",
    help="""
    Mode selection:
    - realtime: PersonaPlex full-duplex (ultra-low latency, interactive)
    - dubbing: Qwen3-TTS narration (multilingual, professional quality)
    - creative: Ollama + Qwen3-TTS (emotional content, storytelling)
    """
)

parser.add_argument(
    "--persona",
    type=str,
    default="You are a helpful assistant.",
    help="Persona/role prompt for PersonaPlex (realtime mode)"
)

parser.add_argument(
    "--voice",
    type=str,
    default="NATF2",
    choices=["NATF0", "NATF1", "NATF2", "NATF3",
             "NATM0", "NATM1", "NATM2", "NATM3",
             "VARF0", "VARF1", "VARF2", "VARF3", "VARF4",
             "VARM0", "VARM1", "VARM2", "VARM3", "VARM4"],
    help="Voice selection for PersonaPlex (realtime mode)"
)

parser.add_argument(
    "--emotion",
    type=str,
    default="",
    help="Emotion instruction for Qwen3-TTS (creative mode)"
)
```

#### Task 1.5: Update `config.py`
**File:** `config.py`
**Section to add:** New mode configurations

**Changes:**
```python
class ApplicationConfig:
    """Application-level configuration."""

    # Mode settings
    DEFAULT_MODE = "realtime"  # Changed from "interactive"
    MODES = ["realtime", "dubbing", "creative"]

    # Real-time mode settings (PersonaPlex)
    REALTIME_VOICE = os.getenv("REALTIME_VOICE", "NATF2")
    REALTIME_PERSONA = os.getenv("REALTIME_PERSONA", "You are a helpful assistant.")
    REALTIME_STREAMING = True
    REALTIME_TIMEOUT = 30.0  # seconds

    # Creative mode settings (Ollama + Qwen3-TTS)
    CREATIVE_LLM = os.getenv("CREATIVE_LLM", "llama3")
    CREATIVE_DEFAULT_EMOTION = os.getenv("CREATIVE_EMOTION", "")
    CREATIVE_TIMEOUT = 120.0  # seconds

    # Dubbing mode settings (unchanged)
    DUBBING_CHUNK_SIZE = 256
```

---

### Phase 2: Documentation Updates (HIGH PRIORITY)

#### Task 2.1: Update `README.md`
**File:** `README.md`

**Changes:**
1. Update project description (lines 1-13)
2. Rewrite Quick Start section with three modes
3. Add mode comparison table
4. Update usage examples
5. Add migration guide from old "interactive" mode

**New Quick Start:**
```markdown
## ⚡ Quick Start

### Real-Time Conversation (PersonaPlex)
Ultra-low latency, full-duplex conversations:
```bash
python3 main.py --mode realtime --persona "You are a tutor" --voice NATF2
```

### High-Fidelity Dubbing (Qwen3-TTS)
Professional narration for content creation:
```bash
python3 main.py --mode dubbing --script "Your narration here"
```

### Creative Assistant (Ollama + Qwen3-TTS)
Emotional storytelling:
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run Velloris
python3 main.py --mode creative --emotion "Speak excitedly"
```
```

#### Task 2.2: Update `ARCHITECTURE.md`
**File:** `ARCHITECTURE.md`

**Changes:**
1. Add section: "Why Three Modes?"
2. Add mode comparison table
3. Update pipeline diagrams
4. Add latency comparison
5. Update PersonaPlex section (currently incorrect)

**New Section:**
```markdown
## Three-Mode Architecture

### Mode Comparison

| Feature | Real-Time | Dubbing | Creative |
|---------|-----------|---------|----------|
| **Engine** | PersonaPlex | Qwen3-TTS | Ollama + Qwen3-TTS |
| **Latency** | 70-170ms ⚡ | N/A | 1-3s |
| **Full-Duplex** | ✅ Yes | ❌ No | ❌ No |
| **Languages** | English | 10 languages | Depends on LLM |
| **Ollama Needed** | ❌ No | ❌ No | ✅ Yes |
| **GPU Required** | Yes (NVIDIA) | Recommended | Recommended |
| **Best For** | Conversations | Narration | Creative content |
```

#### Task 2.3: Update `CLAUDE.md`
**File:** `CLAUDE.md`

**Changes:**
1. Add section on architectural decisions
2. Explain why PersonaPlex was misused
3. Document correct usage patterns
4. Add contributor guidelines for each mode

**New Section:**
```markdown
## Architectural Decisions

### PersonaPlex Usage (Critical)

**WRONG** ❌:
```python
# Don't use PersonaPlex for transcription only!
text = personaplex.transcribe_audio(audio)
```

**CORRECT** ✅:
```python
# Use PersonaPlex for end-to-end S2S
agent_audio = personaplex.process_voice_turn(audio, voice="NATF2", persona="...")
```

PersonaPlex is a complete conversational AI, not an STT engine.
Using it for transcription only wastes 95% of its capabilities.
```

#### Task 2.4: Create `MIGRATION.md`
**New file:** `MIGRATION.md`

**Content:**
```markdown
# Migration Guide: v1.x to v2.0

## Breaking Changes

### Mode Rename: `interactive` → `realtime` or `creative`

**Old Command:**
```bash
python main.py --mode interactive
```

**New Options:**

**Option 1: Real-Time Mode** (Recommended - Faster)
```bash
python main.py --mode realtime --persona "You are a helpful assistant"
```
- Uses PersonaPlex end-to-end
- No Ollama needed
- 10-15x faster latency
- Full-duplex conversations

**Option 2: Creative Mode** (More flexible)
```bash
# Terminal 1
ollama serve

# Terminal 2
python main.py --mode creative --emotion "friendly tone"
```
- Uses Ollama + Qwen3-TTS
- Emotional control
- Multilingual support
- Similar to old interactive

### Summary

| Old | New Equivalent | Reason |
|-----|----------------|--------|
| `--mode interactive` | `--mode realtime` | PersonaPlex S2S (faster) |
| `--mode interactive` | `--mode creative` | Ollama + Qwen3 (flexible) |
| `--mode dubbing` | `--mode dubbing` | No change |
```

---

### Phase 3: Testing & Validation (MEDIUM PRIORITY)

#### Task 3.1: Update Test Suite
**File:** `tests/test_pipeline.py`

**Changes:**
1. Add tests for each mode independently
2. Test PersonaPlex S2S (not just transcription)
3. Test mode routing in orchestrator
4. Test error handling for missing dependencies

**New Tests:**
```python
def test_realtime_mode():
    """Test PersonaPlex end-to-end S2S"""
    orchestrator = LocalVoiceOrchestrator()
    audio = np.random.randn(24000 * 2).astype(np.float32)

    result = orchestrator.route_request(
        mode="realtime",
        audio_input=audio,
        voice_prompt="NATF2.pt",
        text_prompt="You are helpful"
    )

    assert result is not None
    assert len(result) == 2  # (audio, sr)

def test_creative_mode():
    """Test Ollama + Qwen3-TTS pipeline"""
    orchestrator = LocalVoiceOrchestrator()

    result = orchestrator.route_request(
        mode="creative",
        text="Tell me a story",
        emotion="Speak excitedly"
    )

    assert result is not None
```

#### Task 3.2: Integration Testing
**Manual Testing Required:**

1. **Real-Time Mode:**
   - Test on NVIDIA GPU
   - Verify latency < 200ms
   - Test interruption handling
   - Test 16 different voices

2. **Dubbing Mode:**
   - Verify no regressions
   - Test all 10 languages
   - Test voice cloning
   - Test emotion control

3. **Creative Mode:**
   - Test with Ollama running
   - Test without Ollama (should error gracefully)
   - Test emotion instructions
   - Test multilingual generation

---

### Phase 4: Enhancement & Polish (LOW PRIORITY)

#### Task 4.1: Add Examples
**New directory:** `examples/`

**Files to create:**
1. `examples/realtime_conversation.py` - PersonaPlex demo
2. `examples/dubbing_narration.py` - Qwen3-TTS demo
3. `examples/creative_storytelling.py` - Ollama + Qwen3 demo

#### Task 4.2: Performance Optimization
1. Add CPU offload support for PersonaPlex
2. Implement audio streaming for real-time mode
3. Add memory management for mode switching
4. Optimize model loading/unloading

#### Task 4.3: Error Handling
1. Detect missing Ollama gracefully
2. Handle CUDA out-of-memory errors
3. Provide helpful error messages per mode
4. Add fallback strategies

---

## 🎯 Success Criteria

### Must Have (Phase 1-2):
- ✅ PersonaPlex used correctly (end-to-end S2S)
- ✅ Three modes working independently
- ✅ Ollama optional (not required for realtime)
- ✅ Documentation updated completely
- ✅ Migration guide created
- ✅ CLI updated with new modes

### Should Have (Phase 3):
- ✅ All tests passing
- ✅ Integration tests completed
- ✅ Error handling robust
- ✅ Performance benchmarks documented

### Nice to Have (Phase 4):
- ✅ Example scripts for each mode
- ✅ CPU offload support
- ✅ Streaming optimizations
- ✅ Web UI for mode selection

---

## 📊 Expected Impact

### Performance Improvements:
- ⚡ **10-15x faster latency** in realtime mode (170ms vs 2000ms)
- ✅ **Natural interruptions** (95% success rate)
- ✅ **Simpler architecture** (fewer moving parts)

### User Experience:
- 🎭 **More natural conversations** (full-duplex)
- 🎯 **Clear mode separation** (obvious when to use each)
- 📝 **Better documentation** (easier onboarding)
- 🔧 **Easier setup** (Ollama optional)

### Code Quality:
- ♻️ **Proper use of PersonaPlex** (100% capabilities)
- 🏗️ **Cleaner architecture** (mode-based routing)
- 📚 **Better separation of concerns**
- 🧪 **Easier testing** (isolated modes)

---

## ⚠️ Risks & Mitigation

### Risk 1: Breaking Changes for Users
**Impact:** High
**Mitigation:**
- Detailed migration guide
- Clear error messages
- Backward compatibility warnings

### Risk 2: PersonaPlex Integration Complexity
**Impact:** Medium
**Mitigation:**
- Start with stub implementation
- Test thoroughly on CUDA hardware
- Document hardware requirements clearly

### Risk 3: Testing Without Hardware
**Impact:** Medium
**Mitigation:**
- Stub modes for CI/CD
- Manual testing checklist
- Community testing program

---

## 📅 Timeline Estimate

### Phase 1 (Core Refactor): 2-3 days
- Day 1: PersonaPlex + Orchestrator changes
- Day 2: Brain + Main CLI updates
- Day 3: Config + initial testing

### Phase 2 (Documentation): 1-2 days
- Day 4: README, ARCHITECTURE, CLAUDE.md
- Day 5: Migration guide, examples

### Phase 3 (Testing): 1-2 days
- Day 6: Update test suite
- Day 7: Integration testing

### Phase 4 (Polish): 1 day
- Day 8: Examples, error handling, optimization

**Total Estimate: 5-8 days**

---

## 🚀 Next Steps

1. ✅ Review this plan with stakeholders
2. ⏳ Prioritize tasks (use TodoWrite tool)
3. ⏳ Start with Phase 1, Task 1.1 (PersonaPlex refactor)
4. ⏳ Test on CUDA hardware after each major change
5. ⏳ Update documentation incrementally
6. ⏳ Gather community feedback before final release

---

## 📚 References

- [PersonaPlex Research Page](https://research.nvidia.com/labs/adlr/personaplex/)
- [PersonaPlex GitHub](https://github.com/NVIDIA/personaplex)
- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
- [Ollama Documentation](https://ollama.ai)

---

**Document Owner:** Claude
**Last Updated:** 2026-02-05
**Status:** Ready for Implementation
