# Configuration Guide

SonicBoom can be configured via environment variables.

Security-sensitive misconfiguration fails closed: the server refuses to
start rather than falling back to insecure behavior.

Parsing is strict: an absent variable uses its documented default, but a
variable that is present and malformed is a startup failure, never a
silent default. For example `REQUEST_TIMEOUT_SECS=banana` refuses to
start instead of becoming `120`, and `COOKIE_SECURE=treu` refuses to
start instead of silently becoming `false`.

Accepted boolean spellings (all options below): `true`, `false`, `1`,
`0` — textual values are ASCII case-insensitive. `yes`, `no`, typos,
and empty assignments are rejected.

## Environment Variables

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `17842` | Server port |
| `SONICBOOM_ADMIN_ID` | `admin` | Admin panel username (must not be empty) |
| `SONICBOOM_ADMIN_PW` | _(none, required)_ | Admin panel password (min 12 Unicode chars; no defaults/placeholders accepted) |

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_CACHE_DIR` | `./models` | Directory for cached ONNX models |
| `MODEL_REVISION` | pinned SHA (`3cadd1ee…`) | Immutable HuggingFace revision: must be a 40-char commit SHA (never `main`/tags) |
| `MODEL_SHA256_JSON_PATH` | _(unset)_ | Custom complete SHA-256 manifest; **required** when `MODEL_REVISION` differs from the default (may also carry per-file `size` to bound downloads) |
| `HF_TOKEN` | - | HuggingFace token for private models |
| `INFERENCE_STEPS` | `5` | Number of inference steps (1–50) |
| `MODEL_DOWNLOAD_CONNECT_TIMEOUT_SECS` | `10` | Per-file connect timeout for model downloads (1–300) |
| `MODEL_DOWNLOAD_TIMEOUT_SECS` | `1800` | Per-file total timeout for model downloads (1–86400) |

### Security

| Variable | Default | Description |
|----------|---------|-------------|
| `TOKEN_STORE_PATH` | `./tokens.json` | Path to token-hash storage file (created `0600` if missing) |
| `ENABLE_SAMPLE_TOKEN` | `false` | Enable `SAMPLE_TOKEN` for testing (dev only, `1`/`true`) |
| `SONICBOOM_AUTH_REQUIRED` | `true` | Set to `0` or `false` to allow API access without authentication (dev only) |
| `ALLOWED_AUDIO_DIR` | - | **Required with playback:** root directory for queue files (canonicalized, no escapes) |
| `MAX_TEXT_LENGTH` | `10000` | Maximum TTS input length, Unicode chars (1–1000000) |
| `MAX_CHUNK_CHARS` | `200` | Hard maximum text chunk size, Unicode chars (1–10000) |
| `REQUEST_TIMEOUT_SECS` | `120` | Request timeout in seconds (1–3600) |
| `MAX_CONCURRENT_INFERENCE` | `1` | Max simultaneous inferences (1–64) |
| `MAX_PENDING_INFERENCE` | `8` | Max queued (waiting) inferences, overflow → `429` (0–10000) |
| `TTS_RATE_LIMIT_REQUESTS` | `20` | Per-token requests per window for TTS endpoints, `0` disables (0–1000000) |
| `TTS_RATE_LIMIT_WINDOW_SECS` | `60` | Rate-limit window in seconds (1–86400) |
| `TTS_MAX_BODY_BYTES` | `65536` | Body limit for `/api/tts*` (→ `413` when exceeded) (1024–16777216) |
| `OPENAI_MAX_BODY_BYTES` | `65536` | Body limit for `/v1/audio/speech` (1024–16777216) |
| `QUEUE_MAX_BODY_BYTES` | `16384` | Body limit for queue JSON (1024–16777216) |
| `ADMIN_MAX_BODY_BYTES` | `16384` | Body limit for admin forms (1024–16777216) |
| `MAX_PLAYBACK_QUEUE_ITEMS` | `100` | Max waiting playback items, overflow → `429` (1–10000) |
| `TRUST_PROXY` | `false` | Honor forwarded client-IP headers (only with `TRUSTED_PROXIES`) |
| `TRUSTED_PROXIES` | _(empty)_ | Comma-separated proxy IPs/CIDRs; each entry must be a valid IP or CIDR |
| `COOKIE_SECURE` | `false` | Set `true` when serving HTTPS (admin session cookie; required in production HTTPS) |
| `ENABLE_HSTS` | `false` | Send `Strict-Transport-Security` (only when all traffic is known-HTTPS) |
| `ADMIN_SESSION_EXPIRY_SECS` | `28800` | Admin session inactivity expiry, seconds (60–2592000) |
| `TEMP_AUDIO_DIR` | `./temp_audio` | Directory for temporary playback files (created `0700`, files `0600`, `<uuid>.wav` swept on startup) |

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
export PORT=17842
export SONICBOOM_ADMIN_ID=admin
# Generate a real password: python3 -c 'import secrets; print(secrets.token_urlsafe(24))'
export SONICBOOM_ADMIN_PW=<GENERATE-A-RANDOM-PASSWORD>
export TOKEN_STORE_PATH=./tokens.json
export MODEL_CACHE_DIR=./models
export HF_TOKEN=your_huggingface_token
export INFERENCE_STEPS=10
export ALLOWED_AUDIO_DIR=./audio
```

### Windows (PowerShell)

```powershell
$env:PORT=17842
$env:SONICBOOM_ADMIN_ID="admin"
# Generate a real password instead of reusing a placeholder.
$env:SONICBOOM_ADMIN_PW="<GENERATE-A-RANDOM-PASSWORD>"
```

### .env File

Create a `.env` file (gitignored — never commit secrets):

```bash
PORT=17842
SONICBOOM_ADMIN_ID=admin
# Generate a real password: python3 -c 'import secrets; print(secrets.token_urlsafe(24))'
SONICBOOM_ADMIN_PW=<GENERATE-A-RANDOM-PASSWORD>
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
      - "127.0.0.1:17842:17842"
    environment:
      - PORT=17842
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

### Production HTTPS

Terminate TLS at a reverse proxy (nginx, Caddy, Traefik) in front of
SonicBoom and forward plain HTTP to the container. Production checklist:

- `COOKIE_SECURE=true` (required for HTTPS deployments).
- `ENABLE_HSTS=true` only when **all** client traffic is HTTPS.
- The proxy must forward the real client connection; SonicBoom only honors
  `X-Forwarded-For` from `TRUSTED_PROXIES` when `TRUST_PROXY=true`.
  Chains are parsed from the trusted (right) side — configured proxies
  are skipped right-to-left and the first untrusted address is the
  client — so a spoofed leftmost entry can never be selected. Derived
  IPs feed admin-lockout accounting only, never authentication.
- Never expose the plain-HTTP backend publicly: Compose binds loopback
  (`127.0.0.1`) by default; override with `BIND_ADDR=0.0.0.0` only when
  you intentionally need LAN exposure behind your own controls.

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

Model authenticity never relies on trust-on-first-use. Expected SHA-256
digests for all 17 model files at the pinned revision are compiled into
the binary from `models.sha256.json`:

- Every cached file is size-checked against its trusted exact byte size
  and then hashed and compared to its trusted digest before reuse;
  mismatches trigger deletion + redownload.
- Every download is byte-bounded (`Content-Length` must match the trusted
  size, the stream aborts past it and must match it exactly) and the
  temp file's SHA-256 is verified **before** it is renamed into the
  cache — unverified bytes never sit at a final cache path. Persistent
  mismatches fail model preparation instead of loading an unverified
  model.
- Downloads carry explicit connect/total timeouts
  (`MODEL_DOWNLOAD_CONNECT_TIMEOUT_SECS`,
  `MODEL_DOWNLOAD_TIMEOUT_SECS`); failures never log `HF_TOKEN`.
- A custom `MODEL_REVISION` requires `MODEL_SHA256_JSON_PATH` pointing to
  a complete, valid manifest for that exact revision; otherwise startup
  fails. The same policy applies to the reusable `SonicBoomEngine`.
  Custom manifests may use `"path": "<sha256>"` or
  `"path": {"sha256": "<hex>", "size": <bytes>}` per file (one shape
  consistently); sizes enable the byte bounds above.

To move to a newer upstream revision: download all files at the new commit
SHA, compute their SHA-256 digests and exact byte sizes, verify them
independently (e.g. two separate downloads), then supply them via
`MODEL_SHA256_JSON_PATH` together with the new `MODEL_REVISION`.

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
  Hashes must be 64 lowercase hex chars; duplicate ids/hashes are
  rejected.
- Mutations are transactional: the new state is persisted (temp file +
  fsync + atomic rename + parent-directory sync) before live memory is
  swapped, so a failed create/revoke can never leave memory and disk
  disagreeing.
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
unaffected. Admission permits are owned by the blocking inference task
itself, so an HTTP timeout/cancellation can never release a slot while
inference still runs.

### Model Cache

Keep the model cache on fast storage (SSD) for faster loading on restarts.
