import torch
import numpy as np

# Load Silero VAD locally (first time it will download the model)
# Make sure you have 'sounddevice' and 'torchaudio' installed for proper functioning
try:
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
except Exception as e:
    print(
        f"Could not load Silero VAD. Ensure internet connectivity or 'snakers4/silero-vad' is available locally: {e}"
    )

    # Define dummy functions if VAD can't be loaded to prevent crashes
    class DummyVADIterator:
        def __call__(self, *args, **kwargs):
            return {}

        def reset_states(self):
            pass

    VADIterator = DummyVADIterator
    model = None  # Indicate model not loaded


class InterruptionHandler:
    def __init__(self, threshold=0.5):
        if model is None:
            print(
                "WARNING: Silero VAD model not loaded. Interruption detection will not function."
            )
            self.vad_iterator = VADIterator()  # Use dummy
        else:
            self.vad_iterator = VADIterator(model, threshold=threshold)
        self.is_interrupted = False

    def check_for_speech(self, audio_chunk, sampling_rate=16000):
        """
        Processes a 32ms (512 samples) chunk of microphone audio.
        `audio_chunk` should be a numpy array of floats.
        """
        if model is None:  # VAD not loaded
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
        if model is not None:
            self.vad_iterator.reset_states()


if __name__ == "__main__":
    print("InterruptionHandler boilerplate created.")
    # Example usage would require setting up a live audio stream
