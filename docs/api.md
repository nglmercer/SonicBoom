# SonicBoom API Reference

Complete API documentation for SonicBoom TTS server.

## Table of Contents

- [Original TTS API](#original-tts-api)
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

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `voice` | string | No | `M1` | Voice style (M1-M5, F1-F5) |
| `lang` | string | No | `en` | Language code |
| `format` | string | No | `opus` | Output format: `opus`, `wav`, or `mp3` |

**Request Body:** Plain text string

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

### Check Model Status

Check if the model is loaded and ready.

**Endpoint:** `GET /api/status`

**Authentication:** None

**Response:** JSON object with model status

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Model status: `idle`, `downloading`, `loading`, `ready`, or `failed` |
| `progress` | float | Download progress percentage (only when downloading) |
| `error` | string | Error message (only when failed) |

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

## Web Routes

### Health Check

Simple health check endpoint for load balancers and monitoring.

**Endpoint:** `GET /health`

**Authentication:** None

**Response:** Plain text "OK" with status 200

**Example:**

```bash
curl http://localhost:3000/health
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

| Format | Content-Type | Description |
|--------|--------------|-------------|
| `opus` | `audio/opus` | Default Opus/OGG format (recommended) |
| `wav` | `audio/wav` | WAV format (PCM 16-bit) |
| `mp3` | `audio/mpeg` | MP3 format (currently falls back to WAV) |

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

| Status Code | Description |
|-------------|-------------|
| `200` | Success |
| `400` | Bad request (invalid input) |
| `401` | Unauthorized (invalid/missing token) |
| `404` | Not found |
| `422` | Unprocessable entity |
| `500` | Internal server error |
| `503` | Service unavailable (model loading) |

---

## Rate Limiting

The admin panel includes login attempt tracking to prevent brute-force attacks.

- Maximum 5 failed attempts per IP
- 15-minute lockout after threshold exceeded
