# Debugging Report: `main.py --mode dubbing` on macOS

## Problem Summary

The user was unable to run the `dubbing` mode of the Velloris application on their macOS machine with either `cpu` or `mps` devices. The command `python3 main.py --mode dubbing --device cpu` was failing with various errors.

## Investigation

The investigation involved several steps to identify the root cause of the problem:

1.  **Initial Code Analysis:** I started by analyzing the `main.py`, `core/orchestrator.py`, and `engines/qwen_tts.py` files to understand the code flow for the `dubbing` mode. This revealed that the `Qwen3TTSEngine` was responsible for the text-to-speech synthesis.

2.  **Dependency Conflict Identification:** The user mentioned that they had `mlx-lm` and `mlx-vlm` installed, which are used for MLX on Apple Silicon. I discovered a dependency conflict between `qwen-tts` and these libraries:
    -   `qwen-tts` required an older version of the `transformers` library (`<5.0`).
    -   `mlx-lm` and `mlx-vlm` required a newer version of `transformers` (`>=5.0`).

3.  **Attempted Patching:** I initially attempted to patch the `qwen-tts` library to make it compatible with `transformers>=5.0`. This involved fixing several issues related to model configuration and architecture registration. However, this approach proved to be very complex and ultimately unsuccessful.

4.  **Version Downgrade:** The most effective solution was to find a set of compatible library versions. I investigated the release history of `mlx-lm` and `mlx-vlm` and found older versions that were compatible with an older version of `transformers`.

## Solution

The final solution was to downgrade the `transformers`, `mlx-lm`, and `mlx-vlm` libraries to a set of compatible versions.

The following versions were found to be compatible:
-   `transformers==4.57.3`
-   `mlx-lm==0.22.5`
-   `mlx-vlm==0.0.5`

The `requirements.txt` file was updated to pin the `transformers` version:
```
transformers==4.57.3
```

The user then installed the compatible versions of `mlx-lm` and `mlx-vlm`.

## Outcome

After applying the version changes, the `dubbing` mode of the Velloris application now works correctly on macOS with both `cpu` and `mps` devices.
