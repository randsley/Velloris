import sounddevice as sd
import numpy as np
from utils.vad_handler import InterruptionHandler

def play_ai_response(audio_data, handler, samplerate=24000):
    """
    Plays audio data in chunks, with immediate interruption capability.
    `audio_data` is expected to be a numpy array.
    """
    chunk_size = 1024 
    print("AI Speaking...")
    for i in range(0, len(audio_data), chunk_size):
        if handler.is_interrupted:
            print("AI Silenced (Interrupted).")
            sd.stop() # Kill audio output immediately
            break
        
        chunk = audio_data[i:i + chunk_size]
        # Ensure chunk is float32 for sounddevice
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        sd.play(chunk, samplerate=samplerate)
        sd.wait() # Wait for the current chunk to finish playing
    if not handler.is_interrupted:
        print("AI Finished Speaking.")

class IntegratedAudioController:
    def __init__(self, handler: InterruptionHandler):
        self.handler = handler
        self.fs = 16000  # VAD standard rate
        self.output_fs = 24000  # PersonaPlex/Qwen3 rate, though can be resampled
        self.audio_queue = asyncio.Queue() # To buffer AI responses

    async def _input_callback(self, indata, frames, time, status):
        """
        Continuous stream handling Input (Mic).
        """
        if status:
            print("Input callback status:", status)

        # Convert indata to numpy array and check for speech
        audio_chunk = np.frombuffer(indata, dtype=np.int16).astype(np.float32) / 32768.0
        
        if self.handler.check_for_speech(audio_chunk, sampling_rate=self.fs):
            self.handler.is_interrupted = True
            # In a real system, you might also signal the AI to stop generating
            
    async def _output_callback(self, outdata, frames, time, status):
        """
        Continuous stream handling Output (AI Voice).
        """
        if status:
            print("Output callback status:", status)

        if self.handler.is_interrupted:
            outdata.fill(0)  # Immediately mute the speaker buffer
            return

        try:
            # Try to get audio from the queue, non-blocking
            audio_chunk = self.audio_queue.get_nowait()
            outdata[:] = audio_chunk.reshape(outdata.shape) # Fill output buffer
        except asyncio.QueueEmpty:
            outdata.fill(0) # No audio to play, fill with silence

    async def start_session(self):
        print("Starting audio session...")
        # On Mac, 'voice_processing=True' enables hardware Echo Cancellation
        # On Windows, select the 'WASAPI' host API for lowest latency
        # Note: sounddevice callbacks are typically blocking, so for async
        # this might need a dedicated thread or different design.
        # For boilerplate, we'll keep it simple.

        # Find default input and output devices
        default_devices = sd.query_devices(kind='host')
        input_device_info = sd.query_devices(default_devices['default_input_device'])
        output_device_info = sd.query_devices(default_devices['default_output_device'])

        print(f"Using input device: {input_device_info['name']}")
        print(f"Using output device: {output_device_info['name']}")

        with sd.Stream(callback=self._input_callback, channels=1, samplerate=self.fs, blocksize=512,
                       dtype='int16', device=input_device_info['index']) as input_stream, 
             sd.Stream(callback=self._output_callback, channels=1, samplerate=self.output_fs, blocksize=int(self.output_fs * 0.05), # 50ms chunks
                       dtype='float32', device=output_device_info['index']) as output_stream: # Assuming float32 from TTS
            
            print("Session Active. Speak to the Agent...")
            while True: # Keep session alive until explicitly stopped
                await asyncio.sleep(0.1) # Prevent busy-waiting

if __name__ == "__main__":
    print("IntegratedAudioController boilerplate created.")
    # Example usage would require an event loop and a way to feed audio to queue
