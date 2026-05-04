"""
TTS Engine Pool -- Multiple Model Instances for Parallel Generation

Instead of a single Qwen3TTSEngine protected by a lock (which serializes
all generation to one thread), this pool loads N independent model copies.
Each worker thread acquires an engine from the pool, uses it, and returns
it. This enables true parallel GPU generation on MPS.

On a 128GB M4 Max, each model instance is ~8GB. With 20GB headroom,
we can comfortably fit 8 instances and still have room for Whisper + OS.

Usage:
    pool = get_engine_pool()        # Lazy init, loads N models
    engine = pool.acquire()         # Blocks if all engines busy
    try:
        results = engine.generate_chunks(...)
    finally:
        pool.release(engine)        # Return to pool

Or as context manager:
    with pool.engine() as engine:
        results = engine.generate_chunks(...)
"""

import logging
import threading
import queue
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

_pool: Optional["EnginePool"] = None
_pool_lock = threading.Lock()


class EnginePool:
    """Thread-safe pool of Qwen3TTSEngine instances."""

    def __init__(self, pool_size: int = 4):
        self.pool_size = pool_size
        self._engines: queue.Queue = queue.Queue(maxsize=pool_size)
        self._all_engines = []
        self._loaded = False

    def load(self, timeout_per_model: int = 300):
        """Load all model instances into the pool."""
        if self._loaded:
            return

        from src.tts.qwen3_engine import Qwen3TTSEngine
        import psutil

        proc = psutil.Process()
        logger.info(f"Loading TTS engine pool: {self.pool_size} instances...")
        logger.info(f"  RSS before: {proc.memory_info().rss / (1024**3):.2f}GB")

        for i in range(self.pool_size):
            logger.info(f"  Loading engine {i+1}/{self.pool_size}...")
            engine = Qwen3TTSEngine()
            engine._load_model(timeout_seconds=timeout_per_model)
            self._engines.put(engine)
            self._all_engines.append(engine)
            rss = proc.memory_info().rss / (1024**3)
            logger.info(f"  Engine {i+1} loaded (RSS={rss:.2f}GB)")

        self._loaded = True
        rss = proc.memory_info().rss / (1024**3)
        logger.info(
            f"Engine pool ready: {self.pool_size} instances, "
            f"RSS={rss:.2f}GB"
        )

    def acquire(self, timeout: float = 300.0):
        """Acquire an engine from the pool. Blocks if all are in use."""
        try:
            engine = self._engines.get(timeout=timeout)
            return engine
        except queue.Empty:
            raise TimeoutError(
                f"Could not acquire TTS engine from pool within {timeout}s "
                f"(all {self.pool_size} instances busy)"
            )

    def release(self, engine):
        """Return an engine to the pool."""
        self._engines.put(engine)

    @contextmanager
    def engine(self, timeout: float = 300.0):
        """Context manager to acquire and auto-release an engine."""
        eng = self.acquire(timeout)
        try:
            yield eng
        finally:
            self.release(eng)

    def cleanup(self):
        """Free all model resources."""
        for engine in self._all_engines:
            engine.cleanup()
        self._all_engines.clear()
        # Drain the queue
        while not self._engines.empty():
            try:
                self._engines.get_nowait()
            except queue.Empty:
                break
        self._loaded = False
        logger.info("Engine pool cleaned up")

    @property
    def available(self) -> int:
        """Number of engines currently available (not in use)."""
        return self._engines.qsize()

    @property
    def loaded(self) -> bool:
        return self._loaded


def get_engine_pool(pool_size: Optional[int] = None) -> EnginePool:
    """Get or create the global engine pool.

    Args:
        pool_size: Number of model instances. If None, auto-detected
                   based on available memory (8GB per model, 20GB headroom).
    """
    global _pool
    with _pool_lock:
        if _pool is not None and _pool.loaded:
            return _pool

        if pool_size is None:
            import psutil
            avail_gb = psutil.virtual_memory().available / (1024**3)
            model_gb = 8.0
            headroom_gb = 20.0
            pool_size = max(1, min(8, int((avail_gb - headroom_gb) / model_gb)))
            logger.info(
                f"Auto pool size: {pool_size} "
                f"(avail={avail_gb:.0f}GB, model={model_gb}GB, headroom={headroom_gb}GB)"
            )

        _pool = EnginePool(pool_size=pool_size)
        _pool.load()
        return _pool
