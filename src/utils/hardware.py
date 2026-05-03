"""
Hardware Profiler for Resource-Aware Generation

Detects system RAM, GPU capabilities, and computes optimal batch sizes
for TTS generation based on available resources.  Provides runtime memory
monitoring so the worker can adapt under pressure.
"""

import gc
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

import psutil

logger = logging.getLogger(__name__)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Memory diagnostic snapshot
# ---------------------------------------------------------------------------

_BASELINE_RSS: Optional[float] = None  # Set once at first snapshot


def memory_snapshot(label: str = "", log_level: int = logging.INFO) -> dict:
    """
    Log a detailed memory snapshot and return the raw numbers.

    Includes:
      - Process RSS (resident), VMS (virtual)
      - System RAM used / total / available / percent
      - Swap used / total
      - MPS allocator stats (current, peak) when on Apple Silicon
      - Live torch.Tensor count and estimated size
      - Delta vs first-ever snapshot (baseline)
    """
    global _BASELINE_RSS

    proc = psutil.Process()
    mem_info = proc.memory_info()
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()

    rss_gb = mem_info.rss / (1024 ** 3)
    vms_gb = mem_info.vms / (1024 ** 3)
    sys_used_gb = vm.used / (1024 ** 3)
    sys_total_gb = vm.total / (1024 ** 3)
    sys_avail_gb = vm.available / (1024 ** 3)
    swap_used_gb = swap.used / (1024 ** 3)
    swap_total_gb = swap.total / (1024 ** 3)

    if _BASELINE_RSS is None:
        _BASELINE_RSS = rss_gb
    delta_gb = rss_gb - _BASELINE_RSS

    # Torch tensor census
    tensor_count = 0
    tensor_bytes = 0
    mps_current_mb = 0.0
    mps_peak_mb = 0.0
    if _TORCH_AVAILABLE:
        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.Tensor):
                    tensor_count += 1
                    tensor_bytes += obj.nelement() * obj.element_size()
            except (ReferenceError, RuntimeError):
                pass
        tensor_mb = tensor_bytes / (1024 ** 2)

        # MPS allocator stats (Apple Silicon only)
        if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            try:
                mps_current_mb = torch.mps.current_allocated_memory() / (1024 ** 2)
            except Exception:
                pass
        if hasattr(torch, "mps") and hasattr(torch.mps, "driver_allocated_memory"):
            try:
                mps_peak_mb = torch.mps.driver_allocated_memory() / (1024 ** 2)
            except Exception:
                pass
    else:
        tensor_mb = 0.0

    # Python object census (rough)
    gc_counts = gc.get_count()  # (gen0, gen1, gen2)

    snap = {
        "rss_gb": rss_gb,
        "vms_gb": vms_gb,
        "delta_gb": delta_gb,
        "sys_used_gb": sys_used_gb,
        "sys_total_gb": sys_total_gb,
        "sys_avail_gb": sys_avail_gb,
        "sys_percent": vm.percent,
        "swap_used_gb": swap_used_gb,
        "swap_total_gb": swap_total_gb,
        "tensor_count": tensor_count,
        "tensor_mb": tensor_mb,
        "mps_current_mb": mps_current_mb,
        "mps_peak_mb": mps_peak_mb,
        "gc_counts": gc_counts,
    }

    tag = f"[MEM {label}]" if label else "[MEM]"
    logger.log(
        log_level,
        f"{tag}  "
        f"RSS={rss_gb:.2f}GB (delta {'+' if delta_gb >= 0 else ''}{delta_gb:.2f}GB)  |  "
        f"VMS={vms_gb:.2f}GB  |  "
        f"SYS={sys_used_gb:.1f}/{sys_total_gb:.1f}GB ({vm.percent}%)  |  "
        f"SWAP={swap_used_gb:.1f}/{swap_total_gb:.1f}GB  |  "
        f"Tensors={tensor_count} ({tensor_mb:.0f}MB)  |  "
        f"MPS alloc={mps_current_mb:.0f}MB driver={mps_peak_mb:.0f}MB  |  "
        f"GC={gc_counts}"
    )

    return snap

# ---- Constants ----
# Approximate memory footprint of the Qwen3-TTS 1.7B model by dtype
MODEL_MEMORY_GB = {
    "float32": 6.8,   # 4 bytes * 1.7B params
    "bfloat16": 3.4,  # 2 bytes * 1.7B params
    "float16": 3.4,
}

# Conservative estimate for per-segment KV-cache + intermediate tensors
# during batched generation.  Tunable via config.
DEFAULT_PER_SEGMENT_COST_GB = 1.5


@dataclass
class ResourceProfile:
    """Computed resource allocation for a generation run."""

    # Hardware
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0
    gpu_type: str = "cpu"          # "mps", "cuda", "cpu"
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0       # 0 for MPS (unified), actual for CUDA
    cpu_count: int = 1
    device: str = "cpu"
    dtype: str = "float32"

    # Computed settings
    batch_size: int = 1
    max_concurrent_jobs: int = 1
    flush_interval: int = 50       # Flush audio to disk every N segments

    # Config that produced this profile (for logging)
    _notes: list = field(default_factory=list)


def detect_system_memory() -> tuple[float, float]:
    """Return (total_gb, available_gb) of system RAM."""
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)
    available_gb = mem.available / (1024 ** 3)
    return total_gb, available_gb


def detect_gpu() -> dict:
    """Detect GPU type, name, and VRAM."""
    info = {
        "gpu_type": "cpu",
        "gpu_name": "",
        "gpu_vram_gb": 0.0,
        "device": "cpu",
        "dtype": "float32",
    }

    if not _TORCH_AVAILABLE:
        return info

    if torch.cuda.is_available():
        info["gpu_type"] = "cuda"
        info["device"] = "cuda:0"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"] = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        info["dtype"] = "bfloat16"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["gpu_type"] = "mps"
        info["device"] = "mps"
        info["gpu_name"] = "Apple Silicon"
        # MPS uses unified memory -- VRAM = system RAM
        info["gpu_vram_gb"] = 0.0
        info["dtype"] = "float32"  # MPS requires float32

    return info


def get_memory_pressure() -> float:
    """
    Return current memory pressure as a fraction in [0.0, 1.0].
    1.0 means all RAM is consumed.

    Uses psutil.virtual_memory().percent which reflects system-wide
    memory usage including Metal/MPS driver allocations that are
    invisible to per-process RSS and torch.mps.* APIs.
    """
    mem = psutil.virtual_memory()
    return mem.percent / 100.0


def get_system_available_gb() -> float:
    """Return available system RAM in GB.

    This is the most reliable metric on macOS because it captures
    ALL memory consumers including Metal GPU allocations, the
    MPSGraphCache, and MLIR BumpPtrAllocators that are invisible
    to torch.mps.current_allocated_memory() and process RSS.
    """
    mem = psutil.virtual_memory()
    return mem.available / (1024 ** 3)


def check_memory_safe(min_available_gb: float = 16.0) -> tuple[bool, float]:
    """Check if there is enough system memory to continue.

    Args:
        min_available_gb: Minimum available RAM to consider safe.

    Returns:
        (is_safe, available_gb) tuple.
    """
    available = get_system_available_gb()
    return available >= min_available_gb, available


def compute_batch_size(
    available_memory_gb: float,
    model_memory_gb: float,
    per_segment_cost_gb: float = DEFAULT_PER_SEGMENT_COST_GB,
    headroom_fraction: float = 0.20,
    total_memory_gb: float = 0.0,
    min_batch: int = 1,
    max_batch: int = 12,
) -> int:
    """
    Compute the optimal TTS batch size given available resources.

    Formula:
        usable = available_memory - model_memory - (headroom_fraction * total_memory)
        batch  = floor(usable / per_segment_cost)
        result = clamp(batch, min_batch, max_batch)
    """
    headroom_gb = headroom_fraction * total_memory_gb if total_memory_gb > 0 else 0.0
    usable_gb = available_memory_gb - model_memory_gb - headroom_gb

    if usable_gb <= 0:
        logger.warning(
            f"Very low usable memory: {usable_gb:.1f}GB "
            f"(available={available_memory_gb:.1f}, model={model_memory_gb:.1f}, "
            f"headroom={headroom_gb:.1f}). Forcing batch_size=1."
        )
        return min_batch

    batch = int(usable_gb / per_segment_cost_gb)
    result = max(min_batch, min(batch, max_batch))
    return result


def profile_resources(
    batch_size_override: Optional[int] = None,
    memory_headroom_percent: float = 20.0,
    per_segment_cost_gb: float = DEFAULT_PER_SEGMENT_COST_GB,
    min_batch_size: int = 1,
    max_batch_size: int = 12,
    flush_interval: int = 50,
) -> ResourceProfile:
    """
    Build a complete ResourceProfile by detecting hardware and computing
    optimal settings.

    Args:
        batch_size_override: If set to an int, skip auto-detection and use this value.
        memory_headroom_percent: Percentage of total RAM to reserve (0-100).
        per_segment_cost_gb: Estimated GPU memory per segment in a batch.
        min_batch_size: Floor for auto-computed batch size.
        max_batch_size: Ceiling for auto-computed batch size.
        flush_interval: Flush segment audio to disk every N segments.
    """
    total_gb, available_gb = detect_system_memory()
    gpu_info = detect_gpu()
    dtype = gpu_info["dtype"]
    model_gb = MODEL_MEMORY_GB.get(dtype, MODEL_MEMORY_GB["float32"])
    cpu_count = os.cpu_count() or 1

    notes = []

    # For CUDA, use VRAM instead of system RAM for batch calculation
    if gpu_info["gpu_type"] == "cuda" and gpu_info["gpu_vram_gb"] > 0:
        calc_available = gpu_info["gpu_vram_gb"]
        calc_total = gpu_info["gpu_vram_gb"]
        notes.append(f"Using CUDA VRAM ({gpu_info['gpu_vram_gb']:.1f}GB) for batch sizing")
    elif gpu_info["gpu_type"] == "mps":
        # MPS unified memory does NOT benefit from batching like CUDA does.
        # Batched inference multiplies the KV cache linearly with batch size
        # and MPS has no parallel kernel advantage, so batch>1 just wastes
        # memory with no speed gain.  Force sequential generation.
        calc_available = available_gb
        calc_total = total_gb
        max_batch_size = 1
        notes.append(f"MPS detected -- forcing batch_size=1 (batching wastes memory on unified memory)")
    else:
        # CPU fallback
        calc_available = available_gb
        calc_total = total_gb
        notes.append(f"Using system RAM ({total_gb:.1f}GB total, {available_gb:.1f}GB free)")

    # Compute or override batch size
    if batch_size_override is not None and batch_size_override > 0:
        batch_size = max(min_batch_size, min(batch_size_override, max_batch_size))
        notes.append(f"Batch size override: {batch_size_override} -> clamped to {batch_size}")
    else:
        batch_size = compute_batch_size(
            available_memory_gb=calc_available,
            model_memory_gb=model_gb,
            per_segment_cost_gb=per_segment_cost_gb,
            headroom_fraction=memory_headroom_percent / 100.0,
            total_memory_gb=calc_total,
            min_batch=min_batch_size,
            max_batch=max_batch_size,
        )
        notes.append(
            f"Auto batch_size={batch_size} "
            f"(avail={calc_available:.1f}GB - model={model_gb:.1f}GB "
            f"- headroom={calc_total * memory_headroom_percent / 100:.1f}GB) "
            f"/ {per_segment_cost_gb}GB per segment"
        )

    # Max concurrent jobs: conservative -- 1 for most systems, 2 for large RAM
    max_jobs = 1 if total_gb < 64 else 2

    profile = ResourceProfile(
        memory_total_gb=total_gb,
        memory_available_gb=available_gb,
        gpu_type=gpu_info["gpu_type"],
        gpu_name=gpu_info["gpu_name"],
        gpu_vram_gb=gpu_info["gpu_vram_gb"],
        cpu_count=cpu_count,
        device=gpu_info["device"],
        dtype=dtype,
        batch_size=batch_size,
        max_concurrent_jobs=max_jobs,
        flush_interval=flush_interval,
        _notes=notes,
    )

    return profile


def log_profile(profile: ResourceProfile) -> None:
    """Log the resource profile in a human-readable format."""
    logger.info("=" * 60)
    logger.info("RESOURCE PROFILE")
    logger.info("=" * 60)
    logger.info(f"  RAM        : {profile.memory_total_gb:.1f}GB total, {profile.memory_available_gb:.1f}GB available")
    logger.info(f"  GPU        : {profile.gpu_type} ({profile.gpu_name or 'none'})")
    if profile.gpu_vram_gb > 0:
        logger.info(f"  VRAM       : {profile.gpu_vram_gb:.1f}GB")
    logger.info(f"  Device     : {profile.device}  dtype={profile.dtype}")
    logger.info(f"  CPUs       : {profile.cpu_count}")
    logger.info(f"  Batch size : {profile.batch_size}")
    logger.info(f"  Flush every: {profile.flush_interval} segments")
    logger.info(f"  Max jobs   : {profile.max_concurrent_jobs}")
    for note in profile._notes:
        logger.info(f"  [NOTE] {note}")
    logger.info("-" * 60)
