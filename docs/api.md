# SonicBoom API Reference

Complete API documentation for SonicBoom TTS server.

## Table of Contents

- [Original TTS API](#original-tts-api)
- [Audio Queue API](#audio-queue-api)
- [Audio Output Device API](#audio-output-device-api)
- [Web Routes](#web-routes)
- [Authentication](#authentication)
- [Response Formats](#response-formats)
- [Error Codes](#error-codes)

---

## Original TTS API

### Generate TTS Audio

Synthesizes text to speech audio.

**Endpoint:** `POST /api/tts`

**Authentication:** Bearer token required

**Query Parameters:**

| Parameter | Type   | Required | Default | Description                            |
| --------- | ------ | -------- | ------- | -------------------------------------- |
| `voice`   | string | No       | `M1`    | Voice style (M1-M5, F1-F5)             |
| `lang`    | string | No       | `en`    | Language code                          |
| `format`  | string | No       | `opus`  | Output format: `opus`, `wav`, `mp3`, or `flac` (unknown → `400`) |

**Request Body:** Plain text string (max `MAX_TEXT_LENGTH` Unicode chars,
body capped at `TTS_MAX_BODY_BYTES`)

> The built-in web UI (`/`) requires you to paste an API token, which it
> sends as `Authorization: Bearer <token>`. The token stays in the page's
> memory only (never `localStorage`, never logged).

**Response:** Audio data (format based on `format` parameter)

**Example:**

```bash
# Get WAV output
curl -X POST "http://localhost:17842/api/tts?voice=F1&format=wav" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d "Hello, world!" \
  --output audio.wav
```

---

### Synthesize and Play

Synthesizes text and adds it directly to the server's playback queue in a single request.

**Endpoint:** `POST /api/tts/play`

**Authentication:** Bearer token required

**Query Parameters:**

| Parameter  | Type    | Required | Default | Description                            |
| ---------- | ------- | -------- | ------- | -------------------------------------- |
| `voice`    | string  | No       | `M1`    | Voice style (M1-M5, F1-F5)             |
| `lang`     | string  | No       | `en`    | Language code                          |
| `play_now` | boolean | No       | `false` | If `true`, clears queue and plays now  |

**Request Body:** Plain text string

**Response (JSON):**

```json
{
  "success": true,
  "message": "Added to queue",
  "id": "generated-uuid"
}
```

**Example:**

```bash
curl -X POST "http://localhost:17842/api/tts/play?voice=F1&play_now=true" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d "Synthesize and play this immediately on the server speakers."
```

---

### Check Model Status

Check if the model is loaded and ready.

**Endpoint:** `GET /api/status`

**Authentication:** None (public metadata endpoint exposing only
non-sensitive load status; see [OpenAI docs](openai.md#public-metadata-endpoints))

**Response:** JSON object with model status

**Response Fields:**

| Field      | Type   | Description                                                          |
| ---------- | ------ | -------------------------------------------------------------------- |
| `status`   | string | Model status: `idle`, `downloading`, `loading`, `ready`, or `failed` |
| `progress` | float  | Download progress percentage (only when downloading)                 |
| `error`    | string | Error message (only when failed)                                     |

**Example:**

```bash
curl http://localhost:17842/api/status
```

**Example Response:**

```json
{
  "status": "ready",
  "progress": null,
  "error": null
}
```

---

## Audio Queue API

The Audio Queue API allows you to play audio files directly on the server's audio output (e.g., speakers) and manage a playback queue.

### Queue Audio File

Adds an audio file to the playback queue.

**Endpoint:** `POST /api/queue`

**Authentication:** Bearer token required

**Request Body (JSON):**

| Field      | Type    | Required | Description                                                |
| ---------- | ------- | -------- | ---------------------------------------------------------- |
| `path`     | string  | Yes      | Path inside `ALLOWED_AUDIO_DIR` (relative or absolute)     |
| `id`       | string  | No       | Unique identifier 1–128 chars, no control chars (generated if missing) |
| `play_now` | boolean | No       | If `true`, clears the queue and starts playing immediately |

Queue files must live inside the server's `ALLOWED_AUDIO_DIR` (canonicalized;
`..` escapes, symlink escapes, directories, missing files, and non-audio
extensions `.wav/.mp3/.flac/.ogg/.opus` are rejected). All queue endpoints
require bearer authentication. Error responses use `success: false` with an
appropriate status (`400`/`401`/`403`/`429`/`503`) and never echo server
paths. A full playback queue (`MAX_PLAYBACK_QUEUE_ITEMS`) returns `429`;
a dead audio thread returns `503`.

**Example:**

```bash
curl -X POST http://localhost:17842/api/queue \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "path": "./audio/welcome.wav",
    "play_now": false
  }'
```

---

### Control Playback

Control the current playback state.

| Endpoint            | Method | Description                              |
| ------------------- | ------ | ---------------------------------------- |
| `/api/queue/next`   | `POST` | Skip to the next item in the queue       |
| `/api/queue/pause`  | `POST` | Pause current playback                   |
| `/api/queue/resume` | `POST` | Resume paused playback                   |
| `/api/queue/stop`   | `POST` | Stop playback and clear the entire queue |

**Response (JSON):**

```json
{
  "success": true,
  "message": "Playback paused",
  "id": null
}
```

---

### Set Volume

Adjust the master volume for audio playback.

**Endpoint:** `POST /api/queue/volume`

**Request Body (JSON):**

| Field    | Type  | Required | Description                                   |
| -------- | ----- | -------- | --------------------------------------------- |
| `volume` | float | Yes      | Volume level from `0.0` (mute) to `1.0` (max); out-of-range or non-finite values → `400` (never silently clamped) |

**Example:**

```bash
curl -X POST http://localhost:17842/api/queue/volume \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"volume": 0.5}'
```

---

### Get Queue Status

Retrieve information about current playback and the queue.

**Endpoint:** `GET /api/queue/status`

**Response Fields:**

| Field          | Type    | Description                          |
| -------------- | ------- | ------------------------------------ |
| `current`      | object  | Currently playing item (or `null`)   |
| `queue_length` | integer | Number of items waiting in the queue |
| `is_playing`   | boolean | Whether audio is currently playing   |
| `is_paused`    | boolean | Whether audio is currently paused    |
| `volume`       | float   | Current volume level                 |

**Example Response:**

```json
{
  "current": {
    "id": "item-123",
    "path": "/path/to/audio/file.wav"
  },
  "queue_length": 2,
  "is_playing": true,
  "is_paused": false,
  "volume": 0.5
}
```

---

## Audio Output Device API

Playback builds can discover, report, and switch the server's audio
output device at runtime — e.g. speakers, headphones, HDMI, VB-Audio
cables, or loopback devices. HTTP clients cannot enumerate the host's
devices themselves, so SonicBoom owns discovery. All three endpoints
require Bearer [REDACTED] (device names are host metadata) and exist only
when the `playback` feature is enabled (otherwise `404`). No paths or
filesystem data are ever returned.

### List Output Devices

**Endpoint:** `GET /api/audio/devices`

Output devices only (never input-only), with the logical `default` entry
always first. A backend enumeration failure returns `500`, never an
empty success.

**Example Response:**

```json
{
  "devices": [
    {
      "id": "default",
      "name": "System Default",
      "is_default": true,
      "is_selected": false
    },
    {
      "id": "Speakers (Realtek(R) Audio)",
      "name": "Speakers (Realtek(R) Audio)",
      "is_default": false,
      "is_selected": true
    }
  ],
  "selected": "Speakers (Realtek(R) Audio)"
}
```

Device ids are the OS-reported names (backend/OS dependent — CPAL
exposes no stable identifiers). Duplicate names gain ` #2`, ` #3`, ...
suffixes on `id`. `selected` is the configured selection, which may name
a device that is currently missing (it then matches no entry).

### Get Active Output Device

**Endpoint:** `GET /api/audio/output`

Distinguishes the configured selection from the live hardware state, so
a temporarily missing device is visible instead of silently substituted.

**Example Response:**

```json
{
  "device": "CABLE Input (VB-Audio Virtual Cable)",
  "resolved_name": "CABLE Input (VB-Audio Virtual Cable)",
  "available": true
}
```

While the selection is unavailable (missing device, no stream yet),
`available` is `false` and `resolved_name` is `null` — the `device`
selection itself is retained.

### Set Output Device

**Endpoint:** `POST /api/audio/output`

**Request Body (JSON):** `{ "device": "<id or \"default\">" }`

`{"device": "default"}` returns to OS-default behavior. Unknown devices
return `400` with no fallback to another device.

**Example:**

```bash
curl -X POST http://localhost:17842/api/audio/output \
  -H "Authorization: Bearer [REDACTED]" \
  -H "Content-Type: application/json" \
  -d '{"device": "CABLE Input (VB-Audio Virtual Cable)"}'
```

**Example Response:**

```json
{
  "success": true,
  "device": "CABLE Input (VB-Audio Virtual Cable)",
  "resolved_name": "CABLE Input (VB-Audio Virtual Cable)"
}
```

**Switching behavior:** the target is validated against live enumeration
and the new stream is opened before anything is committed — a failed
switch leaves the previous working output active. On success the current
item stops (its temp file is cleaned) and queued items continue on the
new device; the waiting queue is never cleared by a switch. Switching
while paused keeps the queue for `Resume`.

**Disconnect behavior:** a selected device that disappears does not
crash the playback thread and is never auto-switched to another output.
The selection is retained, pending items stay queued, and playback
resumes once the device reappears (immediate retry on commands, throttled
background retry otherwise).

**Persistence:** runtime selection is process-local and does not survive
restarts (SonicBoom never rewrites `.env`). Set `AUDIO_OUTPUT_DEVICE`
for the startup selection.

---

## Web Routes

### Health Check

Liveness endpoint for container `HEALTHCHECK` and monitoring.

**Endpoint:** `GET /health`

**Authentication:** None

**Response:** Plain text "OK" with status 200 (process alive)

**Example:**

```bash
curl http://localhost:17842/health
```

### Readiness Check

Readiness endpoint for load balancers and orchestrators.

**Endpoint:** `GET /ready`

**Authentication:** None

**Response:** `200 ready` when the model is loaded, otherwise `503`
(downloading/loading/idle/failed — no internal details exposed)

**Example:**

```bash
curl http://localhost:17842/ready
```

---

## Authentication

SonicBoom uses Bearer token authentication. Include your API token in the `Authorization` header:

```bash
-H "Authorization: Bearer YOUR_API_TOKEN"
```

The syntax is strict: only `Bearer <token>` is accepted (scheme name is
case-insensitive). Raw tokens without the scheme, `Basic`, `Token`, empty
bearer values, and extra whitespace are all rejected with `401`.

### Optional Authentication

For development or public APIs, you can disable authentication:

```bash
export SONICBOOM_AUTH_REQUIRED=0
```

When disabled, API requests work without any token.

### Getting a Token

1. Access the admin panel at `/admin`
2. Login with admin credentials
3. Navigate to Tokens section
4. Create a new token

### Sample Token

For local development only, you can enable a sample token (the server
logs a prominent warning while it is enabled; never use in production):

```bash
export ENABLE_SAMPLE_TOKEN=1
```

Then use `SAMPLE_TOKEN` for testing.

---

## Response Formats

### Supported Formats

SonicBoom supports multiple audio output formats:

| Format | Content-Type | Description                              |
| ------ | ------------ | ---------------------------------------- |
| `opus` | `audio/ogg; codecs=opus` | Default Opus/OGG format (recommended) |
| `wav`  | `audio/wav`  | WAV format (PCM 16-bit)                  |
| `mp3`  | `audio/mpeg` | MP3 format                               |
| `flac` | `audio/flac` | FLAC format                              |

Unknown formats return `400`; each response is one fully encoded audio
buffer (not an incremental stream).

### Using Format Parameter

**Original API:**

```bash
# Get WAV output
curl -X POST "http://localhost:17842/api/tts?format=wav" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d "Hello, world!" \
  --output audio.wav
```

**OpenAI API:**

```bash
curl -X POST http://localhost:17842/v1/audio/speech \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "voice": "alloy", "response_format": "wav"}' \
  --output audio.wav
```

### Success Response

- **Content-Type:** `audio/ogg; codecs=opus`, `audio/wav`, `audio/mpeg`, or `audio/flac` (based on format)
- **Body:** Raw audio data

### Error Response

- **Content-Type:** `application/json`
- **Body:** Stable error object (never paths, secrets, or internals)

```json
{
  "error": "service_unavailable",
  "message": "Model is downloading (50% complete).",
  "status": 503,
  "request_id": "01234567-89ab-cdef-0123-456789abcdef"
}
```

---

## Error Codes

| Status Code | Description                                         |
| ----------- | --------------------------------------------------- |
| `200`       | Success                                             |
| `400`       | Bad request (invalid input)                         |
| `401`       | Unauthorized (invalid/missing token)                |
| `403`       | Forbidden (queue path outside allowed directory)    |
| `404`       | Not found                                           |
| `413`       | Payload too large (body limit exceeded)             |
| `422`       | Unprocessable entity                                |
| `429`       | Too many requests (rate limit / inference saturated / playback queue full)|
| `500`       | Internal server error (generic message + `request_id`) |
| `503`       | Service unavailable (model loading)                 |

Error responses are stable JSON (`error`, `message`, `status`,
`request_id`) and never include filesystem paths, model internals, or
secrets. Use `request_id` to correlate with server logs.

---

## Rate Limiting

Expensive TTS endpoints (`POST /api/tts`, `POST /api/tts/play`,
`POST /v1/audio/speech`) are rate-limited per API token with a
burst-friendly token bucket: `TTS_RATE_LIMIT_REQUESTS` per
`TTS_RATE_LIMIT_WINDOW_SECS` is the sustained refill rate (default
300/min) and `TTS_RATE_LIMIT_BURST` is the bucket capacity for event
bursts (defaults to the request budget). Set
`TTS_RATE_LIMIT_REQUESTS=0` to disable limiting (not recommended).

Exceeded budgets return `429` with live retry metadata computed from the
actual bucket state:

```http
Retry-After: 12
X-RateLimit-Limit: 300
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 48
```

`Retry-After` is seconds until one request is affordable;
`X-RateLimit-Reset` is seconds until the bucket refills completely. The
JSON body keeps the stable `AppError` shape (`error:
"too_many_requests"`).

SonicBoom returns `429` from three independent protections — never merge
them when debugging:

| Source | Message | Headers |
| ------ | ------- | ------- |
| Token bucket (`RateLimiter`) | `Rate limit exceeded. Try again later.` | `Retry-After` + `X-RateLimit-*` |
| Inference admission (`InferenceGate`) | `Server is busy. Too many pending inference requests.` | none |
| Playback queue bound (`AudioQueue`) | `Playback queue is full. Try again later.` | none |

Inference itself is concurrency-bounded (`MAX_CONCURRENT_INFERENCE`,
`MAX_PENDING_INFERENCE`); saturated requests are rejected with `429`
instead of queueing unboundedly.

The admin panel includes login attempt tracking to prevent brute-force attacks.

- Maximum 5 failed attempts per IP in 10 minutes
- 15-minute lockout after threshold exceeded (expires automatically)
