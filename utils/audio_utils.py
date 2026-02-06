"""
Audio utility functions for resampling, normalization, and chunking.
"""

import numpy as np
import librosa
import torch


def resample_audio(audio, sr_from, sr_to):
    """
    Resample audio to target sample rate.

    Args:
        audio: numpy array or torch tensor of audio
        sr_from: Source sample rate
        sr_to: Target sample rate

    Returns:
        Resampled audio as numpy array
    """
    if sr_from == sr_to:
        return audio

    # Convert to numpy if needed
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()

    # Use librosa for resampling
    if len(audio.shape) == 1:
        # Mono audio
        resampled = librosa.resample(audio, orig_sr=sr_from, target_sr=sr_to)
    else:
        # Multi-channel audio
        resampled = np.zeros(
            (audio.shape[0], int(audio.shape[1] * sr_to / sr_from)), dtype=audio.dtype
        )
        for i in range(audio.shape[0]):
            resampled[i] = librosa.resample(audio[i], orig_sr=sr_from, target_sr=sr_to)

    return resampled.astype(np.float32)


def normalize_audio(audio, target_db=-20.0):
    """
    Normalize audio to target dB level.

    Args:
        audio: numpy array of audio
        target_db: Target loudness in dB (default: -20 dB)

    Returns:
        Normalized audio as numpy array
    """
    # Calculate RMS
    rms = np.sqrt(np.mean(audio**2))

    if rms == 0:
        return audio

    # Calculate target RMS from dB
    target_rms = 10 ** (target_db / 20.0)

    # Normalize
    normalized = audio * (target_rms / rms)

    # Clip to avoid clipping
    normalized = np.clip(normalized, -1.0, 1.0)

    return normalized.astype(np.float32)


def chunk_audio(audio, chunk_size, overlap=0):
    """
    Split audio into overlapping chunks.

    Args:
        audio: numpy array of audio
        chunk_size: Size of each chunk in samples
        overlap: Number of overlapping samples between chunks

    Yields:
        Audio chunks as numpy arrays
    """
    stride = chunk_size - overlap
    for start in range(0, len(audio) - chunk_size + 1, stride):
        yield audio[start : start + chunk_size]

    # Handle remaining samples
    if len(audio) % stride != 0:
        remaining = audio[-(len(audio) % stride) :]
        if len(remaining) > 0:
            yield remaining


def mix_audio(*audio_arrays, normalize=True):
    """
    Mix multiple audio arrays.

    Args:
        audio_arrays: Variable number of audio numpy arrays
        normalize: Whether to normalize output

    Returns:
        Mixed audio as numpy array
    """
    # Pad shorter arrays with zeros
    max_length = max(len(a) for a in audio_arrays)
    mixed = np.zeros(max_length, dtype=np.float32)

    for audio in audio_arrays:
        padded = np.pad(audio, (0, max_length - len(audio)), mode="constant")
        mixed += padded

    if normalize:
        mixed = normalize_audio(mixed)

    return mixed.astype(np.float32)


if __name__ == "__main__":
    # Test functions
    test_audio = np.random.randn(16000).astype(np.float32)

    # Test resample
    resampled = resample_audio(test_audio, 16000, 24000)
    print(
        f"Resample test: {len(test_audio)} samples @ 16kHz -> {len(resampled)} samples @ 24kHz"
    )

    # Test normalize
    normalized = normalize_audio(test_audio, target_db=-20.0)
    print(f"Normalize test: min={normalized.min():.4f}, max={normalized.max():.4f}")

    # Test chunk
    chunks = list(chunk_audio(test_audio, chunk_size=4096, overlap=1024))
    print(f"Chunk test: {len(test_audio)} samples -> {len(chunks)} chunks")
