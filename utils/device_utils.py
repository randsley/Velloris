"""
Device Detection and Platform Optimization

Provides intelligent device detection with platform-specific optimizations
for NVIDIA CUDA, Apple Metal (MPS), and CPU backends.

Supports:
- Auto-detection: CUDA → MPS → CPU priority
- Platform-specific dtype optimization
- FlashAttention 2 capability checking
- Hardware information reporting
"""

import torch
import platform
from typing import Literal, Dict

DeviceType = Literal["cuda", "mps", "cpu", "auto"]


def get_optimal_device(requested: DeviceType = "auto") -> str:
    """
    Detect and validate optimal device for model inference.

    Priority order when using "auto":
    1. CUDA (NVIDIA GPU) - best performance, widest support
    2. MPS (Apple Metal) - native M-series optimization
    3. CPU (fallback) - works everywhere

    Args:
        requested: Requested device type
            - "auto": Auto-detect best available
            - "cuda": NVIDIA GPU (raises error if unavailable)
            - "mps": Apple Metal (raises error if unavailable)
            - "cpu": CPU (always available)

    Returns:
        str: Validated device name ("cuda", "mps", or "cpu")

    Raises:
        ValueError: If requested device is not available
    """
    if requested == "auto":
        # Auto-detect with priority: CUDA → MPS → CPU
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    # Validate requested device
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA requested but not available. "
                "Check NVIDIA GPU installation and torch[cuda] package."
            )
        return "cuda"

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError(
                "MPS (Apple Metal) requested but not available. "
                "Check macOS version (requires 12.3+) and Metal support."
            )
        return "mps"

    if requested == "cpu":
        # CPU is always available
        return "cpu"

    # Invalid device name
    raise ValueError(
        f"Invalid device: {requested}. "
        f"Valid options: 'cuda', 'mps', 'cpu', or 'auto'"
    )


def get_optimal_dtype(device: str) -> torch.dtype:
    """
    Get optimal torch dtype for device.

    Selects dtype based on device capabilities:
    - CUDA: bfloat16 (if supported) or float16 for speed/memory
    - MPS: float32 only (float16/bfloat16 not well-supported)
    - CPU: float32 (standard precision)

    Args:
        device: Device type ("cuda", "mps", or "cpu")

    Returns:
        torch.dtype: Recommended dtype for the device
    """
    if device == "cuda":
        # Check for bfloat16 support (requires Ampere or newer)
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    elif device == "mps":
        # MPS has limited support for reduced precision
        # float32 is most stable
        return torch.float32

    else:  # CPU
        return torch.float32


def validate_flash_attention(device: str) -> bool:
    """
    Check if FlashAttention 2 can be used on device.

    FlashAttention 2 only works on NVIDIA CUDA devices.
    Returns False for MPS (not implemented) and CPU (not applicable).

    Args:
        device: Device type ("cuda", "mps", or "cpu")

    Returns:
        bool: True if FlashAttention 2 is available and usable
    """
    if device != "cuda":
        return False

    # FlashAttention 2 requires CUDA and compatible GPU
    # (Ampere, Hopper, etc.)
    try:
        import flash_attn
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_platform_info() -> Dict[str, any]:
    """
    Get platform and hardware information.

    Returns information about the current system including OS, CPU,
    and GPU availability.

    Returns:
        dict: Platform information containing:
            - os: Operating system name
            - machine: CPU architecture
            - cuda_available: CUDA availability
            - cuda_version: CUDA version if available
            - cuda_device: CUDA device name if available
            - mps_available: Metal Performance Shaders availability
            - python_version: Python version
    """
    info = {
        "os": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }

    # Add CUDA details if available
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = torch.cuda.get_device_capability(0)

    return info


def print_device_info(device: str = "auto"):
    """
    Print formatted device information.

    Args:
        device: Device to get info for ("auto" to auto-detect)
    """
    # Resolve auto device
    if device == "auto":
        device = get_optimal_device("auto")

    platform_info = get_platform_info()

    print("\n🖥️  Device Information")
    print("=" * 50)
    print(f"Platform: {platform_info['os']} ({platform_info['machine']})")
    print(f"Python: {platform_info['python_version']}")

    print(f"\nDetected Hardware:")
    print(f"  CUDA Available: {platform_info['cuda_available']}")
    print(f"  MPS Available: {platform_info['mps_available']}")

    if platform_info["cuda_available"]:
        print(f"  CUDA Version: {platform_info.get('cuda_version', 'Unknown')}")
        print(f"  GPU Device: {platform_info.get('cuda_device', 'Unknown')}")
        print(f"  Compute Capability: {platform_info.get('cuda_capability', 'Unknown')}")

    print(f"\nSelected Device: {device.upper()}")

    # Device-specific info
    if device == "cuda":
        dtype = get_optimal_dtype(device)
        fa2_available = validate_flash_attention(device)
        print(f"  Optimal dtype: {str(dtype).split('.')[-1]}")
        print(f"  FlashAttention 2: {'✅ Available' if fa2_available else '❌ Not available'}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    elif device == "mps":
        print(f"  Optimal dtype: float32")
        print(f"  FlashAttention 2: ❌ Not supported on MPS")
        print(f"  Note: Models will use PyTorch's native MPS ops")

    else:  # CPU
        print(f"  Optimal dtype: float32")
        print(f"  FlashAttention 2: ❌ Not supported on CPU")
        print(f"  Note: CPU inference is slower than GPU")

    print()


if __name__ == "__main__":
    # Test device detection
    print("Testing device detection...\n")

    # Auto-detect
    device = get_optimal_device("auto")
    print(f"Auto-detected device: {device}")

    # Get dtype
    dtype = get_optimal_dtype(device)
    print(f"Optimal dtype: {dtype}")

    # Check FlashAttention
    fa2 = validate_flash_attention(device)
    print(f"FlashAttention 2 available: {fa2}")

    # Print full info
    print_device_info(device)
