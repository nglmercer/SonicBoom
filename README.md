# SonicBoom

A high-performance web server that generates Text-to-Speech (TTS) audio using the Supertonic 3 ONNX model and delivers it via HTTP.

## Documentation Index

This README serves as an index to all SonicBoom documentation:

| Document                                | Description                         |
| --------------------------------------- | ----------------------------------- |
| [API Reference](docs/api.md)            | Complete API documentation          |
| [OpenAI-Compatible API](docs/openai.md) | OpenAI TTS API compatible endpoints |
| [Admin Panel](docs/admin.md)            | Admin panel guide                   |
| [Configuration](docs/config.md)         | TOML, first-run setup and live settings |
| [Security](SECURITY.md)                   | Security policy and credential rotation |

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/nglmercer/SonicBoom.git
cd SonicBoom

# Build the project (headless server, default)
cargo build --release

# Or build with GPU acceleration (enable at most ONE of these per platform):
cargo build --release --features cuda     # NVIDIA CUDA (Linux/Windows)
cargo build --release --features rocm     # AMD ROCm (Linux)
cargo build --release --features coreml   # Apple Silicon CoreML (macOS)

# Or build the desktop system-tray GUI:
cargo build --release --features gui
```

### Run

Desktop builds open a first-run setup wizard. No `.env` file is needed:

```bash
cargo run --release --features gui
```

Choose local or LAN mode, an audio output, model storage, and admin
credentials. Local mode binds loopback without an API token. LAN mode
requires bearer authentication and shows its initial token once.
Configuration lives in the platform application directory as `config.toml`.

For an unattended server, keep using environment variables:

```bash
export PORT=17842
export BIND=127.0.0.1       # use 0.0.0.0 for deliberate LAN exposure
export SONICBOOM_ADMIN_ID=admin
export SONICBOOM_ADMIN_PW=<GENERATE-A-RANDOM-PASSWORD>
export ALLOWED_AUDIO_DIR=./audio
cargo run --release
```

Use a unique password of at least 12 characters. API token hashes remain
in `tokens.json`, which is initialized safely when missing. The admin panel
(`/admin`) displays each newly created token once.

Edit `config.toml` or use `SonicBoom config set` to apply settings live.
See the [configuration guide](docs/config.md) for location overrides,
revisions, migration, source tracking and runtime apply status.

The server will:

1. Start listening on port 17842
2. Download the Supertonic 3 model (first run)
3. Load the model
4. Be ready to serve TTS requests

---

## Features

- **ONNX Runtime Inference** - Supertonic 3 with hardware acceleration (CoreML on Apple Silicon, CUDA/ROCm on Linux via `--features cuda`/`--features rocm`)
- **Buffered Audio Output** - Opus/OGG, WAV, MP3, and FLAC audio output (each response is one fully encoded buffer, not an incremental stream)
- **Token-Based Authentication** - API access control
- **Verified Model Supply Chain** - Pinned revision with compiled-in SHA-256 integrity checks
- **Admin Panel** - Web-based management interface
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI TTS
- **Audio Queue System** - Play audio files directly on the server with queue management
- **Output Device Selection** - Discover and switch server audio outputs (speakers, HDMI, virtual cables) via API
- **Session Management** - Secure admin sessions with lockout protection
- **Desktop Tray (optional)** - System tray GUI via the `gui` feature (macOS/Windows/Linux; Linux uses StatusNotifierItem over D-Bus, no GTK build dependencies)

---

## API Endpoints

### Original TTS API

| Method | Endpoint        | Description                   |
| ------ | --------------- | ----------------------------- |
| `POST` | `/api/tts`      | Generate TTS audio            |
| `POST` | `/api/tts/play` | Synthesize and play on server |
| `GET`  | `/api/status`   | Check model status            |

### Audio Queue API

| Method | Endpoint            | Description                   |
| ------ | ------------------- | ----------------------------- |
| `POST` | `/api/queue`        | Add file to playback queue    |
| `POST` | `/api/queue/next`   | Play next item in queue       |
| `POST` | `/api/queue/pause`  | Pause playback                |
| `POST` | `/api/queue/resume` | Resume playback               |
| `POST` | `/api/queue/stop`   | Stop playback and clear queue |
| `POST` | `/api/queue/volume` | Set playback volume           |
| `GET`  | `/api/queue/status` | Get current queue status      |

### Audio Output Device API (playback builds)

| Method | Endpoint            | Description                        |
| ------ | ------------------- | ---------------------------------- |
| `GET`  | `/api/audio/devices`| List output devices + selection    |
| `GET`  | `/api/audio/output` | Get active output device           |
| `POST` | `/api/audio/output` | Switch output device at runtime    |

### OpenAI-Compatible API

| Method | Endpoint                  | Description                  |
| ------ | ------------------------- | ---------------------------- |
| `POST` | `/v1/audio/speech`        | Generate TTS (OpenAI format) |
| `GET`  | `/v1/models`              | List models                  |
| `GET`  | `/v1/models/list`         | List models (alias)          |
| `GET`  | `/v1/voices`              | List voices                  |

### Admin Panel

| Method   | Endpoint                       | Description      |
| -------- | ------------------------------ | ---------------- |
| `GET`    | `/admin`                       | Admin dashboard  |
| `GET`    | `/admin/login`                 | Login page       |
| `POST`   | `/admin/login`                 | Admin login      |
| `POST`   | `/admin/logout`                | Admin logout     |
| `POST`   | `/admin/tokens`                | Create new token |
| `POST`   | `/admin/tokens/{id}/revoke`    | Revoke token     |

### Web Routes

| Method | Endpoint  | Description                        |
| ------ | --------- | ---------------------------------- |
| `GET`  | `/`       | Home page (TTS demo, needs token)  |
| `GET`  | `/health` | Liveness probe                     |
| `GET`  | `/ready`  | Readiness probe (model loaded)     |

---

## Usage

### Generate TTS Audio

```bash
# Using original API
# Audio is encoded as Opus inside an OGG container (Content-Type: audio/ogg; codecs=opus).
curl -X POST "http://localhost:17842/api/tts?voice=F1" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d "Hello, world!" \
  --output audio.ogg

# Using OpenAI-compatible API
curl -X POST http://localhost:17842/v1/audio/speech \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "voice": "alloy"}' \
  --output audio.ogg
```

---

## Technology Stack

| Component      | Technology                               |
| -------------- | ---------------------------------------- |
| Runtime        | [Tokio](https://tokio.rs/)               |
| Web Framework  | [Axum](https://github.com/tokio-rs/axum) |
| ONNX Inference | [ORT](https://github.com/DBDi/ort)       |
| Audio Encoding | [Opus](https://opus-codec.org/)          |
| Serialization  | [Serde](https://serde.rs/)               |
| Logging        | [Tracing](https://tokio.rs/blog/tracing) |

---

## Project Structure

```textplain
SonicBoom/
├── src/
│   ├── main.rs              # Application entry point
│   ├── config/             # Central ConfigManager, TOML, validation and watcher
│   ├── runtime/            # Subsystems and live reconfiguration
│   ├── server/             # Restartable HTTP listener
│   ├── setup/              # First-run wizard
│   ├── error.rs             # Error types
│   ├── logging.rs           # Logging setup
│   ├── admin/               # Admin panel
│   │   ├── handlers.rs      # Admin HTTP handlers
│   │   ├── lockout.rs       # Failed-login IP lockout
│   │   ├── session.rs       # Admin session helpers
│   │   └── templates.rs     # HTML templates
│   ├── api/                 # API handlers
│   │   ├── tts.rs           # Original TTS API
│   │   ├── openai.rs        # OpenAI-compatible API
│   │   ├── queue.rs         # Audio queue API
│   │   └── audio.rs         # Output device API
│   ├── auth/                # Authentication
│   │   ├── store.rs         # Token storage
│   │   └── token.rs         # Token types/validation
│   ├── tts/                 # TTS engine
│   │   ├── audio.rs         # Audio encoding (Opus/OGG/MP3/FLAC/WAV)
│   │   ├── download.rs      # HuggingFace model download
│   │   ├── inference.rs     # ONNX inference
│   │   ├── model.rs         # Model loading
│   │   ├── queue.rs         # Server-side playback queue
│   │   ├── devices.rs       # Output device discovery
│   │   └── text.rs          # Text normalization
│   └── web/                 # Web frontend
│       └── index.rs         # Home page
├── docs/
│   ├── api.md              # API reference
│   ├── openai.md           # OpenAI API guide
│   ├── admin.md            # Admin panel guide
│   └── config.md           # Configuration guide
├── Cargo.toml
├── Dockerfile
└── docker-compose.yml
```

---

## Docker

```bash
# Build and run with Docker (headless CPU server; no local playback)
docker build -t sonicboom .
docker run -p 127.0.0.1:17842:17842 \
  -e SONICBOOM_ADMIN_ID=admin \
  -e SONICBOOM_ADMIN_PW=<GENERATE-A-RANDOM-PASSWORD> \
  -e HF_TOKEN=your_hf_token \
  -v sonicboom-tokens:/app/data \
  sonicboom

# With local ALSA playback:
docker build --target runtime-playback -t sonicboom:playback .
# With CUDA acceleration (compiles with --features cuda):
docker build --target runtime-cuda -t sonicboom:cuda .
```

Or use docker-compose:

```bash
export SONICBOOM_ADMIN_PW=<GENERATE-A-RANDOM-PASSWORD>
docker compose up -d            # CPU server
docker compose --profile playback up -d   # with local playback
docker compose --profile cuda up -d       # CUDA server
```

Images run as non-root, use digest-pinned bases, and harden the container
(`no-new-privileges`, dropped capabilities, resource limits). See
[Configuration](docs/config.md#container-hardening) for details.

---

## Acknowledgments

- [Supertonic 3](https://huggingface.co/Supertone/supertonic-3) - The TTS model
- [ONNX Runtime](https://onnxruntime.ai/) - Cross-platform ML inference
- Upstream project: [daramkun/SonicBoom](https://github.com/daramkun/SonicBoom)
  (this repository, [nglmercer/SonicBoom](https://github.com/nglmercer/SonicBoom),
  is a hardened fork)
