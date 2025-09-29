"""
Hardware detection and device selection for Whisper Clipboard.
Automatically detects available compute devices (CUDA, ROCm, MPS, CPU).
"""

import torch
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def get_optimal_device() -> str:
    """
    Auto-detect the best available compute device.
    
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
        logger.info(f"CUDA device detected: {device_name}")
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("Apple Metal Performance Shaders (MPS) detected")
        return "mps"
    else:
        logger.info("Using CPU for inference")
        return "cpu"


def get_device_info() -> Tuple[str, dict]:
    """
    Get detailed information about the compute device.
    
    Returns:
        Tuple[str, dict]: Device type and detailed info
    """
    device = get_optimal_device()
    info = {"device": device}
    
    if device == "cuda":
        info.update({
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "memory_allocated": torch.cuda.memory_allocated(0) if torch.cuda.device_count() > 0 else 0,
            "memory_cached": torch.cuda.memory_reserved(0) if torch.cuda.device_count() > 0 else 0,
        })
        
        # Check if this is ROCm (HIP)
        if hasattr(torch.version, 'hip') and torch.version.hip:
            info["hip_version"] = torch.version.hip
            info["backend"] = "ROCm/HIP"
        else:
            info["backend"] = "CUDA"
            
    elif device == "mps":
        info["backend"] = "Apple MPS"
        
    else:
        info["backend"] = "CPU"
        info["cpu_count"] = torch.get_num_threads()
    
    return device, info


def select_model_for_device(device: str, preferred_size: str = "base") -> str:
    """
    Select appropriate Whisper model based on device capabilities.
    
    Args:
        device: Target device ('cuda', 'mps', 'cpu')
        preferred_size: Preferred model size
        
    Returns:
        str: Recommended model size
    """
    if device == "cpu":
        # CPU is limited to smaller models for real-time performance
        if preferred_size in ["large", "medium"]:
            logger.warning(f"Model '{preferred_size}' may be slow on CPU, recommending 'base'")
            return "base"
        return preferred_size if preferred_size in ["tiny", "base", "small"] else "base"
    
    elif device in ["cuda", "mps"]:
        # GPU can handle larger models
        return preferred_size if preferred_size in ["tiny", "base", "small", "medium", "large"] else "base"
    
    return "base"


def check_environment():
    """Check and log environment setup for debugging."""
    import os
    
    logger.info("=== Environment Check ===")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Check ROCm-specific environment variables
    rocm_vars = ['HSA_OVERRIDE_GFX_VERSION', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES']
    for var in rocm_vars:
        value = os.environ.get(var)
        if value:
            logger.info(f"{var}: {value}")
    
    device, info = get_device_info()
    logger.info(f"Selected device: {device}")
    logger.info(f"Device info: {info}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_environment()