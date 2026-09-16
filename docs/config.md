# Configuration Guide

SonicBoom can be configured via environment variables.

Security-sensitive misconfiguration fails closed: the server refuses to
start rather than falling back to insecure behavior.

## Environment Variables

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3000` | Server port |
| `SONICBOOM_ADMIN_ID` | `admin` | Admin panel username (must not be empty) |
| `SONICBOOM_ADMIN_PW` | _(none, required)_ | Admin panel password (min 12 chars; no defaults accepted) |

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_CACHE_DIR` | `./models` | Directory for cached ONNX models |
| `MODEL_REVISION` | pinned SHA (`3cadd1e…`) | Immutable HuggingFace revision for downloads (never `main`) |
| `MODEL_SHA256_JSON_PATH` | _(unset)_ | Optional JSON manifest of expected SHA-256 digests per model file |
| `HF_TOKEN` | - | HuggingFace token for private models |
| `INFERENCE_STEPS` | `5` | Number of inference steps (1–50) |

### Security

| Variable | Default | Description |
|----------|---------|-------------|
| `TOKEN_STORE_PATH` | `./tokens.json` | Path to token-hash storage file (created `0600` if missing) |
| `ENABLE_SAMPLE_TOKEN` | `false` | Enable `SAMPLE_TOKEN` for testing (dev only, `1`/`true`) |
| `SONICBOOM_AUTH_REQUIRED` | `true` | Set to `0` or `false` to allow API access without authentication (dev only) |
| `ALLOWED_AUDIO_DIR` | - | **Required with playback:** root directory for queue files (canonicalized, no escapes) |
| `MAX_TEXT_LENGTH` | `10000` | Maximum TTS input length (Unicode characters) |
| `MAX_CHUNK_CHARS` | `200` | Hard maximum text chunk size (Unicode characters) |
| `REQUEST_TIMEOUT_SECS` | `120` | Request timeout in seconds |
| `MAX_CONCURRENT_INFERENCE` | `1` | Max simultaneous inferences |
| `MAX_PENDING_INFERENCE` | `8` | Max queued (waiting) inferences; overflow → `429` |
| `TTS_RATE_LIMIT_REQUESTS` | `20` | Per-token requests per window for TTS endpoints (`0` disables) |
| `TTS_RATE_LIMIT_WINDOW_SECS` | `60` | Rate-limit window in seconds |
| `TTS_MAX_BODY_BYTES` | `65536` | Body limit for `/api/tts*` (→ `413` when exceeded) |
| `OPENAI_MAX_BODY_BYTES` | `65536` | Body limit for `/v1/audio/speech` |
| `QUEUE_MAX_BODY_BYTES` | `16384` | Body limit for queue JSON |
| `ADMIN_MAX_BODY_BYTES` | `16384` | Body limit for admin forms |
| `TRUST_PROXY` | `false` | Honor forwarded client-IP headers (only with `TRUSTED_PROXIES`) |
| `TRUSTED_PROXIES` | _(empty)_ | Comma-separated proxy IPs/CIDRs trusted for `X-Forwarded-For` |
| `COOKIE_SECURE` | `false` | Set `true` when serving HTTPS (admin session cookie) |
| `ADMIN_SESSION_EXPIRY_SECS` | `28800` | Admin session inactivity expiry (seconds) |
| `TEMP_AUDIO_DIR` | `./temp_audio` | Directory for temporary playback files |

### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_DIR` | `./logs` | Directory for log files |
| `LOG_LEVEL` | `info` | Log level (debug, info, warn, error) |
| `LOG_TO_FILE` | `true` | Enable file logging |
| `LOG_TO_STDOUT` | `true` | Enable console logging |

> Logging hygiene: bearer tokens, admin passwords, `HF_TOKEN`, session
> cookies, and CSRF secrets are never logged. Request bodies (private TTS
> text) are not logged. Error responses carry a `request_id` for
> server-side correlation without leaking internals.

---

## Setting Environment Variables

### Linux/macOS

```bash
export PORT=3000
export SONICBOOM_ADMIN_ID=admin
export SONICBOOM_ADMIN_PW=your_strong_password_12_plus_chars
export TOKEN_STORE_PATH=./tokens.json
export MODEL_CACHE_DIR=./models
export HF_TOKEN=your_huggingface_token
export INFERENCE_STEPS=10
export ALLOWED_AUDIO_DIR=./audio
```

### Windows (PowerShell)

```powershell
$env:PORT=3000
$env:SONICBOOM_ADMIN_ID="admin"
$env:SONICBOOM_ADMIN_PW="your_strong_password_12_plus_chars"
```

### .env File

Create a `.env` file (gitignored — never commit secrets):

```bash
PORT=3000
SONICBOOM_ADMIN_ID=admin
SONICBOOM_ADMIN_PW=your_strong_password_12_plus_chars
TOKEN_STORE_PATH=./tokens.json
MODEL_CACHE_DIR=./models
HF_TOKEN=your_huggingface_token
INFERENCE_STEPS=10
ALLOWED_AUDIO_DIR=./audio
```

Restrict permissions: `chmod 600 .env tokens.json`.

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
      - SONICBOOM_ADMIN_PW=${SONICBOOM_ADMIN_PW:?required}
      - HF_TOKEN=your_hf_token
      - INFERENCE_STEPS=10
    volumes:
      - ./models:/app/models
      - tokendata:/app/data
```

Token data lives in a named volume, so a missing local `tokens.json` is
handled safely. For a local (non-Docker) run, create it with:

```bash
printf '[]\n' > tokens.json
chmod 600 tokens.json
```

### Container Hardening

The Compose file and images follow production hardening practices:

- Containers run as non-root (`sonicboom`, uid 10001)
- `no-new-privileges:true`, `cap_drop: [ALL]`
- CPU/memory resource limits
- `tmpfs` for `/tmp`; writable mounts limited to
  `/app/models`, `/app/logs`, `/app/data`, `/app/temp_audio`
- Digest-pinned base images; separate CPU / playback / CUDA targets

For stricter setups, add a read-only root filesystem with tmpfs mounts for
the writable paths, and terminate TLS at a reverse proxy setting
`COOKIE_SECURE=true`.

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

Models are automatically downloaded from HuggingFace on first run, pinned to
the immutable `MODEL_REVISION` commit (never the mutable `main` branch):

```
2026-03-11T20:45:24.127580Z  INFO SonicBoom::tts::download: Downloading: onnx/duration_predictor.onnx
```

Once downloaded, they're cached locally in `MODEL_CACHE_DIR`.

### Integrity Verification

- Downloads are written atomically (temp file + fsync + rename) and each
  file's SHA-256 is recorded in a `.sha256` sidecar.
- Cached files are re-verified on startup (sidecar digest, or the
  `MODEL_SHA256_JSON_PATH` manifest when provided). Mismatches trigger
  deletion + redownload; persistent mismatches fail safely.
- For strongest supply-chain guarantees, generate a manifest after a
  trusted first download (see `models.sha256.example.json`) and set
  `MODEL_SHA256_JSON_PATH` in production.

---

## Token Storage

### tokens.json Format

Tokens are stored as a JSON array of **hash** records — raw bearer values
are never persisted:

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "token_hash": "<64-hex-sha256>",
    "created_at": "2026-03-11T12:00:00Z",
    "expires_at": null,
    "revoked": false
  }
]
```

**Fields:**

- `id` - Unique identifier (UUID v4)
- `token_hash` - SHA-256 hex of the bearer token (lookup key)
- `created_at` - Creation timestamp (ISO 8601)
- `expires_at` - Optional expiration timestamp (ISO 8601 or null)
- `revoked` - Whether the token has been revoked

**Rules:**

- A missing file is created safely as `[]` with `0600` permissions.
- A malformed/unreadable file **fails startup** — it is never silently
  replaced with an empty store and never overwritten automatically.
- Writes are atomic (temp file + fsync + rename).
- See `tokens.example.json` for the initial-file template.

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
Values are capped at 50; larger values are rejected at startup.

### Inference Concurrency

Model inference is effectively serialized. `MAX_CONCURRENT_INFERENCE` (1)
and `MAX_PENDING_INFERENCE` (8) bound simultaneous and queued work;
overflow returns `429`. Tune up only with threading headroom — quality is
unaffected.

### Model Cache

Keep the model cache on fast storage (SSD) for faster loading on restarts.
