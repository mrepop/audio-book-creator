#!/usr/bin/env python3
"""
Measure actual system memory consumption per generate_custom_voice call.

This script generates a fixed number of TTS calls and measures:
  1. psutil.virtual_memory().available (system-wide, includes Metal driver)
  2. Process RSS (what our logs show)
  3. torch.mps.driver_allocated_memory() (what PyTorch reports)

This gives us ground truth about how much memory each call actually
consumes at the OS level, not just what PyTorch thinks.

Usage:
    venv/bin/python scripts/measure_memory_leak.py --calls 20

After each call, it also tests whether cleanup() + empty_cache()
actually reclaims anything at the OS level.
"""

import argparse
import gc
import os
import subprocess
import threading
import time
import psutil
import numpy as np


def get_process_mem_from_os() -> float:
    """Get process memory in GB using ps (what Activity Monitor shows)."""
    pid = os.getpid()
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)],
            text=True,
        ).strip()
        return int(out) / (1024 * 1024)  # ps reports KB -> GB
    except Exception:
        return 0.0


def get_system_mem():
    """Return (available_gb, used_gb, psutil_rss_gb, os_rss_gb, mps_driver_mb)."""
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    psutil_rss = proc.memory_info().rss / (1024**3)
    os_rss = get_process_mem_from_os()
    
    mps_driver = 0.0
    try:
        import torch
        if hasattr(torch.mps, "driver_allocated_memory"):
            mps_driver = torch.mps.driver_allocated_memory() / (1024**2)
    except Exception:
        pass
    
    return vm.available / (1024**3), vm.used / (1024**3), psutil_rss, os_rss, mps_driver


class MemoryMonitor:
    """Background thread that samples process memory every second via ps."""
    def __init__(self):
        self.peak_gb = 0.0
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            mem = get_process_mem_from_os()
            self.samples.append((time.time(), mem))
            if mem > self.peak_gb:
                self.peak_gb = mem
            self._stop.wait(1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=10, help="Number of generate calls")
    parser.add_argument("--cycle-every", type=int, default=0, help="Cycle model every N calls (0=never)")
    args = parser.parse_args()

    import torch
    from qwen_tts import Qwen3TTSModel

    print(f"PyTorch {torch.__version__} | MPS available: {torch.backends.mps.is_available()}")
    print(f"Generating {args.calls} calls, cycle every {args.cycle_every or 'never'}")
    print()

    # Start background memory monitor
    monitor = MemoryMonitor()
    monitor.start()

    # Baseline before model load
    avail0, used0, psutil_rss0, os_rss0, mps0 = get_system_mem()
    print(f"BASELINE: avail={avail0:.1f}GB  used={used0:.1f}GB  psutil_RSS={psutil_rss0:.2f}GB  ps_RSS={os_rss0:.2f}GB  MPS_drv={mps0:.0f}MB")

    # Load model
    print("Loading model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="mps",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    
    avail1, used1, psutil_rss1, os_rss1, mps1 = get_system_mem()
    print(f"POST-LOAD: avail={avail1:.1f}GB  used={used1:.1f}GB  psutil_RSS={psutil_rss1:.2f}GB  ps_RSS={os_rss1:.2f}GB  MPS_drv={mps1:.0f}MB")
    print(f"  Model cost: avail={avail0-avail1:.1f}GB  psutil_RSS={psutil_rss1-psutil_rss0:.2f}GB  ps_RSS={os_rss1-os_rss0:.2f}GB")
    print()

    texts = [
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning, there was nothing but darkness and silence.",
        "She walked slowly through the garden, admiring each flower.",
        "To be or not to be, that is the question.",
        "It was a bright cold day in April, and the clocks were striking thirteen.",
        "Call me Ishmael.",
        "The sun rose slowly over the mountains, casting long shadows across the valley below.",
        "He said nothing.",
        "All happy families are alike; each unhappy family is unhappy in its own way.",
    ]

    print(f"{'Call':>4}  {'ps_RSS':>8}  {'psutil':>8}  {'SysAvail':>8}  {'MPS_drv':>8}  {'Peak_ps':>8}  {'Time':>6}  Notes")
    print("-" * 95)

    for i in range(args.calls):
        text = texts[i % len(texts)]

        # Record peak before this call
        peak_before = monitor.peak_gb
        t0 = time.perf_counter()
        
        with torch.inference_mode():
            wavs, sr = model.generate_custom_voice(
                text=text,
                language="English",
                speaker="Ryan",
                temperature=0.7,
                max_new_tokens=2048,
            )
        
        audio = wavs[0]
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().float().numpy().copy()
        del wavs
        gc.collect()
        gc.collect()
        torch.mps.empty_cache()
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        del audio
        
        elapsed = time.perf_counter() - t0
        avail, used, psutil_rss, os_rss, mps_drv = get_system_mem()
        peak_during = monitor.peak_gb
        
        notes = ""
        
        if args.cycle_every > 0 and (i + 1) % args.cycle_every == 0:
            pre_cycle_rss = os_rss
            del model
            gc.collect()
            gc.collect()
            torch.mps.empty_cache()
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            
            post_del_rss = get_process_mem_from_os()
            
            model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                device_map="mps",
                dtype=torch.float32,
                attn_implementation="eager",
            )
            avail, used, psutil_rss, os_rss, mps_drv = get_system_mem()
            notes = f"CYCLED: ps_RSS {pre_cycle_rss:.1f}->{post_del_rss:.1f}->{os_rss:.1f}GB"
        
        print(
            f"{i+1:4d}  {os_rss:7.2f}GB  {psutil_rss:7.2f}GB  {avail:7.1f}GB  {mps_drv:7.0f}MB  "
            f"{peak_during:7.2f}GB  {elapsed:5.1f}s  {notes}"
        )

    monitor.stop()

    # Final cleanup
    pre_del = get_process_mem_from_os()
    del model
    gc.collect()
    gc.collect()
    torch.mps.empty_cache()
    if hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    post_del = get_process_mem_from_os()
    
    avail_final, used_final, psutil_rss_final, os_rss_final, mps_final = get_system_mem()
    print()
    print(f"FINAL (after model deleted):")
    print(f"  ps_RSS: {pre_del:.2f}GB -> {post_del:.2f}GB -> {os_rss_final:.2f}GB")
    print(f"  psutil_RSS: {psutil_rss_final:.2f}GB")
    print(f"  SysAvail: {avail_final:.1f}GB  MPS_drv: {mps_final:.0f}MB")
    print(f"  Peak process memory (ps): {monitor.peak_gb:.2f}GB")
    print(f"  Total system memory consumed: {avail0 - avail_final:.1f}GB (vs baseline)")


if __name__ == "__main__":
    main()
