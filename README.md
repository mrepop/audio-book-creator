# Audio Book Creator

Convert ebooks into high-quality audiobooks with ultra-realistic, context-aware voices using Qwen3 TTS.

## Features

- **Ultra-Realistic Voices**: Qwen3-TTS generates voices indistinguishable from human narration
- **Seamless Audio**: No chopping, no clipping -- clear cadence, tempo, and vocal patterns via crossfade concatenation
- **Context-Aware Speech**: NLP-driven emotion detection, emphasis, pacing, and tone analysis for each segment
- **Web Interface**: Full control over character voices, sentence-level editing, and generation parameters
- **Deterministic Voices**: Consistent character voices across the entire book via seed-based generation and voice profiles
- **Dramatis Personae**: Auto-generated character sheet with inferred voice traits -- easy customization per character
- **Hands-Free Mode**: Fully automated audiobook generation with best-effort voice assignment and context analysis

## Architecture

```
audio-book-creator/
  src/
    api/           # FastAPI application, routes, schemas, config
    models/        # SQLAlchemy database models
    ebook/         # Ebook parsing (EPUB, PDF, TXT)
    nlp/           # Character detection, dialogue parsing, context analysis
    tts/           # Qwen3 TTS engine, voice profile management
    audio/         # Prosody analysis, concatenation, normalization
    workers/       # Background job processing
    services/      # Business logic layer
  frontend/        # React + TypeScript web UI (planned)
  storage/         # Books, voices, outputs, temp files
  tests/           # Unit and integration tests
  config.yaml      # Application configuration
```

## Technology Stack

- **Backend**: Python 3.12, FastAPI, SQLAlchemy
- **TTS**: Qwen3-TTS-12Hz-1.7B-Base (via qwen-tts)
- **NLP**: spaCy (character detection, context analysis)
- **Audio**: librosa, soundfile, pydub, numpy
- **Ebook**: ebooklib (EPUB), PyPDF2 (PDF), plain text
- **Frontend**: React + TypeScript (planned)
- **Database**: SQLite (dev) / PostgreSQL (prod)

## Requirements

- Python 3.10-3.12
- 16GB+ RAM recommended
- GPU optional but recommended (CUDA/MPS) for TTS generation
- 10GB+ storage for TTS models

## Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/mrepop/audio-book-creator.git
cd audio-book-creator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 3. Start the server

```bash
python -m src.api.main
```

### 4. Open the API

- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## API Endpoints

### Books
- `POST /api/books/upload` - Upload an ebook (EPUB, PDF, TXT)
- `GET /api/books/` - List all books
- `GET /api/books/{id}` - Get book details with chapters
- `POST /api/books/{id}/parse` - Parse book into chapters
- `POST /api/books/{id}/analyze` - Run NLP analysis (character detection, context)
- `GET /api/books/{id}/dramatis-personae` - Get character sheet

### Characters
- `GET /api/characters/book/{book_id}` - List characters for a book
- `PATCH /api/characters/{id}` - Update character details/voice assignment

### Voice Profiles
- `POST /api/voices/` - Create voice profile
- `GET /api/voices/` - List all voice profiles
- `PATCH /api/voices/{id}` - Update voice settings
- `POST /api/voices/{id}/reference-audio` - Upload reference audio

### Generation
- `POST /api/generation/jobs` - Create audiobook generation job
- `GET /api/generation/jobs/{id}` - Get job status/progress
- `POST /api/generation/jobs/{id}/cancel` - Cancel a job
- `GET /api/generation/jobs/{id}/download` - Download completed audiobook

### Segment Editing
- `GET /api/generation/chapters/{id}/segments` - List segments for editing
- `PATCH /api/generation/segments/{id}` - Update segment text/emphasis/voice
- `POST /api/generation/segments/{id}/regenerate` - Regenerate single segment audio

## Configuration

All settings are in `config.yaml`. Key sections:

- **server**: Host, port, reload settings
- **tts**: Qwen3 model, temperature, sampling parameters
- **audio**: Sample rates, crossfade, silence durations, normalization
- **nlp**: spaCy model, context analysis toggles
- **voices**: Narrator defaults, deterministic seed settings
- **generation**: Auto-mode settings, job concurrency

Override via environment variables:
```bash
export SERVER_PORT=9000
export LOG_LEVEL=DEBUG
export TTS_ENGINE=qwen3
```

## How It Works

1. **Upload**: User uploads an ebook (EPUB, PDF, or TXT)
2. **Parse**: System extracts chapters and identifies structure
3. **Analyze**: NLP engine detects characters, parses dialogue, analyzes context (emotion, emphasis, pacing)
4. **Dramatis Personae**: Character sheet generated with inferred voice traits
5. **Customize** (optional): User tweaks character voices, edits segments
6. **Generate**: Qwen3 TTS produces audio segment-by-segment with context-aware parameters
7. **Assemble**: Segments concatenated with crossfading, normalized, exported as MP3/M4B/WAV

## Project Status

### Phase 1: Foundation [COMPLETE]
- Project structure and configuration
- Database models (Book, Chapter, Character, VoiceProfile, Segment, GenerationJob)
- FastAPI application with full CRUD API
- Ebook parsing (EPUB, TXT)
- NLP module (character detection, context analysis, dialogue parsing)
- Qwen3 TTS engine wrapper with voice profiles
- Audio processing (prosody, concatenation, normalization)

### Phase 2: Generation Pipeline [PLANNED]
- Full end-to-end generation pipeline
- Hands-free automated mode
- Progress tracking and checkpointing
- Chapter-by-chapter generation with voice consistency

### Phase 3: Web Interface [PLANNED]
- React + TypeScript frontend
- Book upload and management UI
- Dramatis personae editor
- Sentence-level editing
- Audio player with chapter navigation

## License

MIT
