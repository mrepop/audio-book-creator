"""
LLM Context Analyzer

Uses a local LLM (Qwen2.5-Instruct via mlx-lm) to analyze text context
and generate per-sentence vocal directions for TTS.

The LLM runs in a separate subprocess (venv-llm/) to avoid dependency
conflicts between mlx-lm and qwen-tts.
"""

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

from .sentence_splitter import Sentence

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
LLM_VENV_PYTHON = str(PROJECT_ROOT / "venv-llm" / "bin" / "python3")
LLM_WORKER_SCRIPT = str(PROJECT_ROOT / "src" / "nlp" / "llm_worker.py")

# Default model -- 7B for speed, 14B for quality
DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


@dataclass
class SentenceAnnotation:
    """LLM-generated annotation for a single sentence."""
    sentence_index: int
    speaker: str = "Narrator"
    emotion: str = "neutral"
    emphasis: str = "normal"
    pacing: str = "normal"
    vocal_direction: str = ""


class LLMContextAnalyzer:
    """
    Manages an LLM subprocess for context-aware vocal direction generation.

    The LLM is loaded once and kept alive for the duration of the analysis.
    Requests are sent as JSON over stdin, responses read from stdout.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, window_size: int = 12, overlap: int = 3):
        self._model_name = model_name
        self._window_size = window_size
        self._overlap = overlap
        self._process: Optional[subprocess.Popen] = None

    def start(self):
        """Start the LLM worker subprocess."""
        if self._process is not None and self._process.poll() is None:
            return  # Already running

        logger.info(f"Starting LLM worker: {self._model_name}")
        self._process = subprocess.Popen(
            [LLM_VENV_PYTHON, LLM_WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line-buffered
        )

        # Send model name
        self._send({"model": self._model_name})

        # Wait for ready signal (model loading can take 10-30s)
        logger.info("Waiting for LLM model to load...")
        response = self._recv(timeout=120)
        if response is None:
            # Check if process died
            if self._process.poll() is not None:
                stderr = self._process.stderr.read() if self._process.stderr else ""
                raise RuntimeError(f"LLM worker died during startup: {stderr[-500:]}")
            raise RuntimeError("LLM worker did not respond within 120s")

        if "error" in response:
            raise RuntimeError(f"LLM worker startup failed: {response['error']}")

        logger.info("LLM worker ready")

    def stop(self):
        """Stop the LLM worker subprocess."""
        if self._process is None:
            return
        try:
            self._send({"command": "quit"})
            self._process.wait(timeout=10)
        except Exception:
            self._process.kill()
        finally:
            self._process = None
            logger.info("LLM worker stopped")

    def analyze_sentences(
        self,
        sentences: List[Sentence],
        character_names: List[str] = None,
    ) -> List[SentenceAnnotation]:
        """
        Analyze all sentences using sliding context windows.

        Args:
            sentences: List of Sentence objects from the sentence splitter
            character_names: Known character names for speaker identification

        Returns:
            List of SentenceAnnotation objects, one per input sentence
        """
        if not sentences:
            return []

        if self._process is None or self._process.poll() is not None:
            self.start()

        annotations: Dict[int, SentenceAnnotation] = {}
        total_windows = max(1, (len(sentences) - self._overlap) // (self._window_size - self._overlap) + 1)

        logger.info(
            f"Analyzing {len(sentences)} sentences in ~{total_windows} windows "
            f"(size={self._window_size}, overlap={self._overlap})"
        )

        window_idx = 0
        i = 0
        while i < len(sentences):
            window_end = min(i + self._window_size, len(sentences))
            window_sents = sentences[i:window_end]

            # Build context from preceding sentences (not in the window)
            context_start = max(0, i - 5)
            context_sents = sentences[context_start:i]
            context_text = " ".join(s.text for s in context_sents) if context_sents else ""

            # Send request to LLM worker
            request = {
                "sentences": [
                    {"text": s.text, "is_dialogue": s.is_dialogue, "index": s.index}
                    for s in window_sents
                ],
                "context": context_text,
                "characters": character_names or [],
            }

            t0 = time.perf_counter()
            self._send(request)
            response = self._recv(timeout=180)
            elapsed = time.perf_counter() - t0

            if response and "annotations" in response:
                raw_annotations = response["annotations"]
                # Map annotations back to sentence indices
                for j, ann in enumerate(raw_annotations):
                    if j < len(window_sents):
                        sent_idx = window_sents[j].index
                        # Only update if not already annotated (first window wins for overlap)
                        if sent_idx not in annotations:
                            annotations[sent_idx] = SentenceAnnotation(
                                sentence_index=sent_idx,
                                speaker=ann.get("speaker", "Narrator"),
                                emotion=ann.get("emotion", "neutral"),
                                emphasis=ann.get("emphasis", "normal"),
                                pacing=ann.get("pacing", "normal"),
                                vocal_direction=ann.get("vocal_direction", ""),
                            )

                logger.info(
                    f"  Window {window_idx+1}/{total_windows}: "
                    f"{len(raw_annotations)} annotations in {elapsed:.1f}s"
                )
            else:
                error = response.get("error", "Unknown error") if response else "No response"
                logger.warning(
                    f"  Window {window_idx+1}/{total_windows} failed: {error}. "
                    f"Using fallback for {len(window_sents)} sentences."
                )
                # Fallback: create neutral annotations
                for s in window_sents:
                    if s.index not in annotations:
                        annotations[s.index] = SentenceAnnotation(
                            sentence_index=s.index,
                            speaker="Narrator",
                            emotion="neutral",
                            emphasis="normal",
                            pacing="normal",
                            vocal_direction="Read in a clear, steady narrative voice.",
                        )

            # Advance window
            window_idx += 1
            i += self._window_size - self._overlap

        # Return in order, with fallbacks for any missed sentences
        result = []
        for sent in sentences:
            if sent.index in annotations:
                result.append(annotations[sent.index])
            else:
                result.append(SentenceAnnotation(
                    sentence_index=sent.index,
                    vocal_direction="Read in a natural, clear voice.",
                ))
        return result

    def _send(self, data: dict):
        """Send a JSON message to the worker."""
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("LLM worker not running")
        self._process.stdin.write(json.dumps(data) + "\n")
        self._process.stdin.flush()

    def _recv(self, timeout: float = 60) -> Optional[dict]:
        """Read a JSON response from the worker."""
        if self._process is None or self._process.stdout is None:
            return None

        import select
        # Use select for timeout on stdout
        ready, _, _ = select.select([self._process.stdout], [], [], timeout)
        if not ready:
            return None

        line = self._process.stdout.readline()
        if not line:
            return None

        try:
            return json.loads(line.strip())
        except json.JSONDecodeError:
            logger.warning(f"Non-JSON from worker: {line[:200]}")
            return None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __del__(self):
        self.stop()
