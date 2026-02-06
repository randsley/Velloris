import torch
import numpy as np

# Silero VAD module-level variables (lazy loaded)
_vad_model = None
_vad_utils = None
_vad_loaded = False


def _load_silero_vad():
    """Load Silero VAD from local path or torch.hub (lazy loading)."""
    global _vad_model, _vad_utils, _vad_loaded

    if _vad_loaded:
        return _vad_model, _vad_utils

    # Lazy import to avoid circular dependency
    from config import Config

    local_path = Config.model.SILERO_VAD_DIR
    local_model_file = local_path / "silero_vad.jit"

    # Try local model first
    if local_model_file.exists():
        print(f"Loading Silero VAD from local path: {local_model_file}")
        try:
            _vad_model = torch.jit.load(str(local_model_file))
            # Get utils from torch.hub (they're just Python functions)
            _, _vad_utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            _vad_loaded = True
            return _vad_model, _vad_utils
        except Exception as e:
            print(f"Failed to load local Silero VAD: {e}")

    # Check offline mode
    if Config.model.OFFLINE_MODE:
        print("[X] Offline mode enabled but Silero VAD not found locally")
        print(f"  Expected path: {local_model_file}")
        print("  Run: python download_models.py --model vad")
        _vad_loaded = True
        return None, None

    # Fall back to torch.hub download
    print("Loading Silero VAD from torch.hub...")
    try:
        _vad_model, _vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        _vad_loaded = True
        return _vad_model, _vad_utils
    except Exception as e:
        print(f"Failed to load Silero VAD: {e}")
        _vad_loaded = True
        return None, None


class DummyVADIterator:
    """Dummy VAD iterator when Silero VAD is not available."""

    def __call__(self, *args, **kwargs):
        return {}

    def reset_states(self):
        pass


class InterruptionHandler:
    """
    Voice Activity Detection handler for interruption detection.

    Uses Silero VAD for detecting speech in audio chunks.
    VAD model is loaded lazily on first instantiation.
    """

    def __init__(self, threshold=0.5):
        # Load VAD model lazily
        model, utils = _load_silero_vad()

        self._model = model
        self.is_interrupted = False

        if model is None:
            print(
                "WARNING: Silero VAD model not loaded. Interruption detection will not function."
            )
            self.vad_iterator = DummyVADIterator()
        else:
            # Extract VADIterator from utils
            (
                get_speech_timestamps,
                save_audio,
                read_audio,
                VADIterator,
                collect_chunks,
            ) = utils
            self.vad_iterator = VADIterator(model, threshold=threshold)

    def check_for_speech(self, audio_chunk, sampling_rate=16000):
        """
        Processes a 32ms (512 samples) chunk of microphone audio.
        `audio_chunk` should be a numpy array of floats.
        """
        if self._model is None:  # VAD not loaded
            return False

        # Ensure audio_chunk is float32 and 16000Hz expected by Silero VAD
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Silero VAD expects 16kHz audio, if input is different, resample might be needed
        # For simplicity in boilerplate, assuming 16kHz input.
        # If your sounddevice stream uses a different rate, you'll need `torchaudio.transforms.Resample`

        speech_dict = self.vad_iterator(audio_chunk, return_seconds=True)
        if speech_dict:
            # If 'start' is in the dictionary, speech has begun
            if "start" in speech_dict:
                print("!!! INTERRUPTION DETECTED !!!")
                self.is_interrupted = True
                return True
        return False

    def reset(self):
        self.is_interrupted = False
        if self._model is not None:
            self.vad_iterator.reset_states()


if __name__ == "__main__":
    print("InterruptionHandler boilerplate created.")
    # Example usage would require setting up a live audio stream
