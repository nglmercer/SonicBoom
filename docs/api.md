# SonicBoom API Reference

Complete API documentation for SonicBoom TTS server.

## Table of Contents

- [Original TTS API](#original-tts-api)
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

**Request Body:** Plain text string

**Response:** Audio data (Opus/OGG format)

**Example:**

```bash
curl -X POST "http://localhost:3000/api/tts?voice=F1&lang=en" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d "Hello, world!" \
  --output audio.ogg
```

---

### Check Model Status

Check if the model is loaded and ready.

**Endpoint:** `GET /api/status`

**Authentication:** None

**Response:** JSON object with model status

**Example:**

```bash
curl http://localhost:3000/api/status
```

---

## Authentication

SonicBoom uses Bearer token authentication. Include your API token in the `Authorization` header:

```bash
-H "Authorization: Bearer YOUR_API_TOKEN"
```

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

### Success Response

- **Content-Type:** `audio/opus`
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
