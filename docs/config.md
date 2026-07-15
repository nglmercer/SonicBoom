# Configuration Guide

SonicBoom can be configured via environment variables.

## Environment Variables

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3000` | Server port |
| `SONICBOOM_ADMIN_ID` | `admin` | Admin panel username |
| `SONICBOOM_ADMIN_PW` | `1234` | Admin panel password |

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_CACHE_DIR` | `./models` | Directory for cached ONNX models |
| `HF_TOKEN` | - | HuggingFace token for private models |
| `INFERENCE_STEPS` | `5` | Number of inference steps |

### Security

| Variable | Default | Description |
|----------|---------|-------------|
| `TOKEN_STORE_PATH` | `./tokens.json` | Path to token storage file |
| `ENABLE_SAMPLE_TOKEN` | `false` | Enable `SAMPLE_TOKEN` for testing |
| `SONICBOOM_AUTH_REQUIRED` | `true` | Set to `0` or `false` to allow API access without authentication |
| `ALLOWED_AUDIO_DIR` | - | Restrict audio queue file access to this directory (prevents path traversal) |
| `MAX_TEXT_LENGTH` | `10000` | Maximum character length for TTS input text |
| `REQUEST_TIMEOUT_SECS` | `120` | Request timeout in seconds |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_DIR` | `./logs` | Directory for log files |
| `LOG_LEVEL` | `info` | Log level (debug, info, warn, error) |
| `LOG_TO_FILE` | `true` | Enable file logging |
| `LOG_TO_STDOUT` | `true` | Enable console logging |

---

## Setting Environment Variables

### Linux/macOS

```bash
export PORT=3000
export SONICBOOM_ADMIN_ID=admin
export SONICBOOM_ADMIN_PW=your_secure_password
export TOKEN_STORE_PATH=./tokens.json
export MODEL_CACHE_DIR=./models
export HF_TOKEN=your_huggingface_token
export INFERENCE_STEPS=10
export ENABLE_SAMPLE_TOKEN=1
```

### Windows (PowerShell)

```powershell
$env:PORT=3000
$env:SONICBOOM_ADMIN_ID="admin"
$env:SONICBOOM_ADMIN_PW="your_secure_password"
```

### .env File

Create a `.env` file:

```bash
PORT=3000
SONICBOOM_ADMIN_ID=admin
SONICBOOM_ADMIN_PW=your_secure_password
TOKEN_STORE_PATH=./tokens.json
MODEL_CACHE_DIR=./models
HF_TOKEN=your_huggingface_token
INFERENCE_STEPS=10
ENABLE_SAMPLE_TOKEN=1
```

---

## Docker Configuration

### Environment Variables

```yaml
# docker-compose.yml
services:
  sonicboom:
    image: sonicboom
    ports:
      - "3000:3000"
    environment:
      - PORT=3000
      - SONICBOOM_ADMIN_ID=admin
      - SONICBOOM_ADMIN_PW=your_password
      - HF_TOKEN=your_hf_token
      - INFERENCE_STEPS=10
    volumes:
      - ./models:/app/models
      - ./tokens.json:/app/tokens.json
```

---

## Model Cache

### Directory Structure

```
models/
├── onnx/
│   ├── duration_predictor.onnx
│   ├── text_encoder.onnx
│   ├── vector_estimator.onnx
│   ├── vocoder.onnx
│   ├── unicode_indexer.json
│   └── tts.json
├── config.json
└── voice_styles/
    ├── M1.json
    ├── M2.json
    ├── M3.json
    ├── M4.json
    ├── M5.json
    ├── F1.json
    ├── F2.json
    ├── F3.json
    ├── F4.json
    └── F5.json
```

### Downloading Models

Models are automatically downloaded from HuggingFace on first run:

```
2026-03-11T20:45:24.127580Z  INFO SonicBoom::tts::download: Downloading: onnx/duration_predictor.onnx
```

Once downloaded, they're cached locally in `MODEL_CACHE_DIR`.

---

## Token Storage

### tokens.json Format

Tokens are stored as a JSON array of token objects:

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "value": "a1b2c3d4e5f6...",
    "created_at": "2026-03-11T12:00:00Z",
    "expires_at": null,
    "revoked": false
  }
]
```

**Fields:**
- `id` - Unique identifier (UUID v4)
- `value` - The token string used for authentication
- `created_at` - Creation timestamp (ISO 8601)
- `expires_at` - Optional expiration timestamp (ISO 8601 or null)
- `revoked` - Whether the token has been revoked

---

## Performance Tuning

### Inference Steps

Higher values = better quality but slower synthesis:

| Steps | Quality | Speed |
|-------|---------|-------|
| 1-3 | Low | Fast |
| 5 | Medium | Normal |
| 10 | High | Slow |
| 20+ | Very High | Very Slow |

**Recommendation:** Start with `5` and adjust based on your quality/speed needs.

### Model Cache

Keep the model cache on fast storage (SSD) for faster loading on restarts.
