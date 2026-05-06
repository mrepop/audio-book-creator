# Audio Book Creator

# Just a heads up, this is under very heavy development and it's not even close to finished, if you try and use this, you know, godspeed, at your own risk and all. I doubt it'll do anything seriously destructive, but it most likely won't create what you're looking for, maybe not even words at this point.  


Convert ebooks into full audiobooks using Qwen3-TTS with character voice assignment, sentence-level chunking, and production-quality audio splicing.

## Features

### Core Pipeline
- EPUB parsing with automatic title/author metadata extraction
- NLP analysis: character detection via spaCy NER, dialogue attribution, gender inference, auto voice profile creation
- Sentence-level chunking: splits text at sentence boundaries, targets 12-15 second chunks for optimal Qwen3 quality
- Qwen3-TTS CustomVoice integration with 9 preset speakers (Ryan, Aiden, Vivian, Serena, etc.)
- Instruction-based emotion control: emotion, emphasis, and pacing per segment via natural language instruct parameter
- Hann-window crossfade splicing with per-chunk LUFS normalization for seamless audio joins
- Chapter assembly from disk-based segment audio files
- Full audiobook concatenation with chapter silence gaps

### Generation Studio (Frontend)
- Three generation modes:
  - Auto: one-click full audiobook generation with defaults
  - Chapter: select and generate individual chapters with preview
  - Sentence: drill down to individual segments with editable emotion/emphasis/pacing, inline audio preview, and regeneration
- Job management: pause, resume, cancel with real-time progress tracking
- Crash recovery: orphaned GENERATING jobs auto-marked as FAILED on restart, resumable
- Segment preview: generate and play audio for a single segment via inline audio player

### Memory Management
- Patched PyTorch MPS graph cache: locally applied PyTorch PR #181485 to clear MPSGraphCache on empty_cache(), fixing the unbounded Metal driver memory leak
- System-level memory guard: checks psutil.virtual_memory().available before each generate call, cycles engine if available RAM drops below 16GB, fails gracefully below 8GB
- Singleton TTS engine: model loaded once, shared across all worker threads and resume calls
- Disk-based chapter assembly: segment audio written to WAV files immediately, not held in memory arrays
- Per-batch empty_cache() + synchronize(): clears MPS graph cache between segment batches

### Concurrency and Reliability
- Per-job cancellation tokens via threading.Event -- pause/cancel signal delivered immediately
- Stop checks between every segment -- responds to pause within seconds
- Metal command buffer serialization via _generate_lock -- prevents concurrent MPS submissions from crashing
- 40 concurrent jobs supported -- GPU work serialized through lock, CPU work (chunking, splicing, I/O) runs in parallel

### Audio Quality
- repetition_penalty=1.05 prevents Qwen3 stuttering/looping artifacts
- max_new_tokens capped per chunk based on estimated duration (12Hz * 2.5x headroom) -- prevents runaway generation producing garbage
- Temperature 0.7 for voice consistency across chunks
- Deterministic seeds with per-chunk offset for reproducibility with variety

## Tech Stack

- Backend: Python 3.12, FastAPI, SQLAlchemy, SQLite
- Frontend: React 18, TypeScript, Vite, TailwindCSS, Lucide icons
- TTS: Qwen3-TTS-12Hz-1.7B-CustomVoice via qwen-tts package
- Audio: soundfile, numpy, librosa (resampling)
- ML: PyTorch 2.11 (locally patched for MPS graph cache fix)
- NLP: spaCy (en_core_web_sm)

## Requirements

- Python 3.12+
- macOS with Apple Silicon (MPS) or Linux with CUDA
- 16GB+ RAM (128GB recommended for full book generation)
- ~8GB for model weights

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/mrepop/audio-book-creator.git
cd audio-book-creator
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Install frontend dependencies
cd frontend && npm install && cd ..

# 3. Start both servers
bash scripts/dev.sh

# 4. Open http://localhost:5173
```

## Project Structure

```
audio-book-creator/
  src/
    api/              - FastAPI app, routes, config, database
      main.py         - Server startup, logging, crash recovery
      config.py       - YAML config loader with ChunkingConfig
      routes/         - books, characters, voices, generation endpoints
      schemas.py      - Pydantic models
    audio/            - Audio processing
      splicer.py      - Hann crossfade + LUFS normalization
      concatenator.py
      normalizer.py
    tts/              - TTS engine
      qwen3_engine.py - Qwen3 wrapper with generate_chunks, MPS lock
      chunker.py      - Sentence-level text splitter
    workers/          - Background job processing
      generation_worker.py - Main pipeline with batching + cleanup
    nlp/              - NLP analysis
      character_detector.py
      dialogue_parser.py
      context_analyzer.py
    ebook/            - Book parsing
      epub_parser.py
    models/           - SQLAlchemy models
      database.py     - TZDateTime, JobStatus, all models
    utils/
      hardware.py     - Memory diagnostics, system memory guard
  frontend/
    src/
      pages/          - LibraryPage, BookDetailPage, GenerationPage, VoiceStudioPage
      lib/api.ts      - Typed API client
      App.tsx          - Router + layout
  tests/
    unit/             - test_chunker, test_splicer, test_hardware, test_resume, test_tz_datetime
  scripts/
    dev.sh            - Start backend + frontend
    measure_memory_leak.py - OS-level memory measurement tool
  config.yaml         - All configuration
  storage/            - Books, voices, outputs, temp files
```

## Configuration

All settings in config.yaml:

- chunking: target_chunk_seconds=12, max_chunk_seconds=15, crossfade_ms=100, sentence_silence_ms=200, paragraph_silence_ms=500
- resources: batch_size=1 (MPS), memory_pressure_threshold=0.85
- generation: max_concurrent_jobs=40, checkpoint_interval=10
- tts.qwen3: default_temperature=0.7, repetition_penalty=1.05, max_new_tokens=2048
- audio: sample_rate=24000, normalization_target_db=-16

## Known Issues

- PyTorch MPS graph cache: requires locally patched PyTorch (PR #181485) to prevent unbounded memory growth. Without the patch, empty_cache() does not clear the MPSGraphCache and memory grows ~300MB per generate call. The patch source is at /tmp/pytorch-patch.
- Runaway generation: some texts cause Qwen3 to generate far longer audio than expected. Mitigated by capping max_new_tokens per chunk, but occasional long outputs still occur.
- Single GPU serialization: all TTS calls serialized through one lock on MPS since Metal command queues are not thread-safe. True parallelism requires multiple GPUs or CPU fallback.
- Character detection accuracy: NER sometimes misidentifies places/objects as characters. Gender inference uses pronoun proximity which can be inaccurate for minor characters.
- Frontend preview playback: the Chapter and Sentence mode preview buttons may not work if a generation job is actively running (GPU lock contention).

## API Endpoints

- POST /api/books/upload - Upload EPUB/PDF/TXT
- POST /api/books/{id}/parse - Parse into chapters
- POST /api/books/{id}/analyze - NLP character detection
- POST /api/generation/jobs - Start generation job
- POST /api/generation/jobs/{id}/pause - Pause job
- POST /api/generation/jobs/{id}/resume - Resume job
- POST /api/generation/jobs/{id}/cancel - Cancel job
- GET /api/generation/jobs/{id}/download - Download audiobook
- POST /api/generation/segments/{id}/preview - Preview segment audio
- PATCH /api/generation/segments/{id} - Update segment params
- GET /api/generation/chapters/{id}/segments - List chapter segments

## Development

```bash
# Run tests
venv/bin/python -m pytest tests/ -v

# Memory leak measurement
venv/bin/python scripts/measure_memory_leak.py --calls 20
```

## License

Proprietary - Seventh Dominion
