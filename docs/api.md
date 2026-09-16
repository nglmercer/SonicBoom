# SonicBoom API Reference

Complete API documentation for SonicBoom TTS server.

## Table of Contents

- [Original TTS API](#original-tts-api)
- [Audio Queue API](#audio-queue-api)
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
| `format`  | string | No       | `opus`  | Output format: `opus`, `wav`, `mp3`, or `flac` |

**Request Body:** Plain text string (max `MAX_TEXT_LENGTH` Unicode chars,
body capped at `TTS_MAX_BODY_BYTES`)

> The built-in web UI (`/`) requires you to paste an API token, which it
> sends as `Authorization: Bearer <token>`. The token stays in the page's
> memory only (never `localStorage`, never logged).

**Response:** Audio data (format based on `format` parameter)

**Example:**

```bash
# Get WAV output
curl -X POST "http://localhost:3000/api/tts?voice=F1&format=wav" \
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
curl -X POST "http://localhost:3000/api/tts/play?voice=F1&play_now=true" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d "Synthesize and play this immediately on the server speakers."
```

---

### Check Model Status

Check if the model is loaded and ready.

**Endpoint:** `GET /api/status`

**Authentication:** None

**Response:** JSON object with model status

**Response Fields:**

| Field      | Type   | Description                                                          |
| ---------- | ------ | -------------------------------------------------------------------- |
| `status`   | string | Model status: `idle`, `downloading`, `loading`, `ready`, or `failed` |
| `progress` | float  | Download progress percentage (only when downloading)                 |
| `error`    | string | Error message (only when failed)                                     |

**Example:**

```bash
curl http://localhost:3000/api/status
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
appropriate status (`400`/`401`/`403`/`503`) and never echo server paths.

**Example:**

```bash
curl -X POST http://localhost:3000/api/queue \
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
| `volume` | float | Yes      | Volume level from `0.0` (mute) to `1.0` (max) |

**Example:**

```bash
curl -X POST http://localhost:3000/api/queue/volume \
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

## Web Routes

### Health Check

Liveness endpoint for container `HEALTHCHECK` and monitoring.

**Endpoint:** `GET /health`

**Authentication:** None

**Response:** Plain text "OK" with status 200 (process alive)

**Example:**

```bash
curl http://localhost:3000/health
```

### Readiness Check

Readiness endpoint for load balancers and orchestrators.

**Endpoint:** `GET /ready`

**Authentication:** None

**Response:** `200 ready` when the model is loaded, otherwise `503`
(downloading/loading/idle/failed — no internal details exposed)

**Example:**

```bash
curl http://localhost:3000/ready
```

---

## Authentication

SonicBoom uses Bearer token authentication. Include your API token in the `Authorization` header:

```bash
-H "Authorization: Bearer YOUR_API_TOKEN"
```

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

For development, you can enable a sample token:

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
| `opus` | `audio/opus` | Default Opus/OGG format (recommended)    |
| `wav`  | `audio/wav`  | WAV format (PCM 16-bit)                  |
| `mp3`  | `audio/mpeg` | MP3 format (currently falls back to WAV) |

### Using Format Parameter

**Original API:**

```bash
# Get WAV output
curl -X POST "http://localhost:3000/api/tts?format=wav" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d "Hello, world!" \
  --output audio.wav
```

**OpenAI API:**

```bash
curl -X POST http://localhost:3000/v1/audio/speech \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "voice": "alloy", "response_format": "wav"}' \
  --output audio.wav
```

### Success Response

- **Content-Type:** `audio/opus`, `audio/wav`, or `audio/mpeg` (based on format)
- **Body:** Raw audio data

### Error Response

- **Content-Type:** `application/json`
- **Body:** Error message

```json
{
  "error": "Model is downloading (50% complete)."
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
| `429`       | Too many requests (rate limit / inference saturated)|
| `500`       | Internal server error (generic message + `request_id`) |
| `503`       | Service unavailable (model loading)                 |

Error responses are stable JSON (`error`, `message`, `status`,
`request_id`) and never include filesystem paths, model internals, or
secrets. Use `request_id` to correlate with server logs.

---

## Rate Limiting

Expensive TTS endpoints (`POST /api/tts`, `POST /api/tts/play`,
`POST /v1/audio/speech`) are rate-limited per API token
(`TTS_RATE_LIMIT_REQUESTS` per `TTS_RATE_LIMIT_WINDOW_SECS`, default
20/min; exceeded → `429`).

Inference itself is concurrency-bounded (`MAX_CONCURRENT_INFERENCE`,
`MAX_PENDING_INFERENCE`); saturated requests are rejected with `429`
instead of queueing unboundedly.

The admin panel includes login attempt tracking to prevent brute-force attacks.

- Maximum 5 failed attempts per IP in 10 minutes
- 15-minute lockout after threshold exceeded (expires automatically)
