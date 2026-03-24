# SonicBoom

A high-performance web server that generates Text-to-Speech (TTS) audio using the Supertonic 2 ONNX model and delivers it via HTTP.

## Documentation Index

This README serves as an index to all SonicBoom documentation:

| Document                                | Description                         |
| --------------------------------------- | ----------------------------------- |
| [API Reference](docs/api.md)            | Complete API documentation          |
| [OpenAI-Compatible API](docs/openai.md) | OpenAI TTS API compatible endpoints |
| [Admin Panel](docs/admin.md)            | Admin panel guide                   |
| [Configuration](docs/config.md)         | Environment variables and settings  |

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/daramkun/SonicBoom.git
cd SonicBoom

# Build the project
cargo build --release
```

### Run

```bash
# Set environment variables (optional)
export PORT=3000
export SONICBOOM_ADMIN_ID=admin
export SONICBOOM_ADMIN_PW=your_secure_password

# Run the server
cargo run --release
```

The server will:

1. Start listening on port 3000
2. Download the Supertonic 2 model (first run)
3. Load the model
4. Be ready to serve TTS requests

---

## Features

- **ONNX Runtime Inference** - Supertonic 2 with CoreML acceleration on Apple Silicon
- **Streaming Audio Output** - Real-time Opus/OGG encoding
- **Token-Based Authentication** - API access control
- **Admin Panel** - Web-based management interface
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI TTS
- **Audio Queue System** - Play audio files directly on the server with queue management
- **Session Management** - Secure admin sessions with lockout protection

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

### OpenAI-Compatible API

| Method | Endpoint           | Description                  |
| ------ | ------------------ | ---------------------------- |
| `POST` | `/v1/audio/speech` | Generate TTS (OpenAI format) |
| `GET`  | `/v1/models`       | List models                  |
| `GET`  | `/v1/voices`       | List voices                  |

### Admin Panel

| Method   | Endpoint            | Description      |
| -------- | ------------------- | ---------------- |
| `GET`    | `/admin`            | Admin dashboard  |
| `POST`   | `/admin/login`      | Admin login      |
| `POST`   | `/admin/logout`     | Admin logout     |
| `GET`    | `/admin/tokens`     | List API tokens  |
| `POST`   | `/admin/tokens`     | Create new token |
| `DELETE` | `/admin/tokens/:id` | Revoke token     |

### Web Routes

| Method | Endpoint  | Description  |
| ------ | --------- | ------------ |
| `GET`  | `/`       | Home page    |
| `GET`  | `/health` | Health check |

---

## Usage

### Generate TTS Audio

```bash
# Using original API
curl -X POST "http://localhost:3000/api/tts?voice=F1" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d "Hello, world!" \
  --output audio.ogg

# Using OpenAI-compatible API
curl -X POST http://localhost:3000/v1/audio/speech \
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
│   ├── config.rs            # Configuration management
│   ├── error.rs             # Error types
│   ├── admin/               # Admin panel
│   ├── api/                 # API handlers
│   │   ├── tts.rs          # Original TTS API
│   │   └── openai.rs       # OpenAI-compatible API
│   ├── auth/               # Authentication
│   ├── tts/                # TTS engine
│   └── web/                # Web frontend
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
# Build and run with Docker
docker build -t sonicboom .
docker run -p 3000:3000 \
  -e SONICBOOM_ADMIN_ID=admin \
  -e SONICBOOM_ADMIN_PW=password \
  -e HF_TOKEN=your_hf_token \
  sonicboom
```

Or use docker-compose:

```bash
docker-compose up -d
```

---

## Acknowledgments

- [Supertonic 2](https://huggingface.co/Supertone/supertonic-2) - The TTS model
- [ONNX Runtime](https://onnxruntime.ai/) - Cross-platform ML inference
