# OpenAI-Compatible API

SonicBoom provides OpenAI TTS API-compatible endpoints, allowing you to use it as a drop-in replacement for OpenAI's TTS service.

## Overview

The OpenAI-compatible API follows the same request/response format as OpenAI's `/v1/audio/speech` endpoint. This allows existing applications using OpenAI TTS to switch to SonicBoom with minimal code changes.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/audio/speech` | Generate TTS audio |
| `GET` | `/v1/models` | List available models |
| `GET` | `/v1/models/list` | List available models (alias) |
| `GET` | `/v1/voices` | List available voices |

---

## Generate Speech

### POST /v1/audio/speech

Synthesizes text to speech using OpenAI-compatible request format.

**Authentication:** Bearer token required

**Request Body:**

```json
{
  "model": "tts-1",
  "input": "Hello, world!",
  "voice": "alloy",
  "response_format": "opus",
  "speed": 1.0
}
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | No | `tts-1` | Model ID (ignored, Supertonic 2 used) |
| `input` | string | Yes | - | Text to synthesize |
| `voice` | string | No | `alloy` | Voice to use |
| `response_format` | string | No | `opus` | Output format (opus, mp3, wav, aac, flac) |
| `speed` | number | No | `1.0` | Speech speed (not implemented) |

**Response:** Audio data in specified format

**Example:**

```bash
curl -X POST http://localhost:3000/v1/audio/speech \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello, world!",
    "voice": "alloy",
    "response_format": "opus"
  }' \
  --output audio.ogg
```

---

## Voice Mapping

SonicBoom maps OpenAI voice names to Supertonic 2 voice styles:

| OpenAI Voice | Supertonic 2 | Description |
|--------------|---------------|-------------|
| `alloy` | M1 | Male voice 1 |
| `echo` | M2 | Male voice 2 |
| `fable` | M3 | Male voice 3 |
| `onyx` | M4 | Male voice 4 |
| `nova` | F1 | Female voice 1 |
| `shimmer` | F2 | Female voice 2 |

### Direct Voice Names

You can also use Supertonic 2 voice names directly:

| Voice | Gender |
|-------|--------|
| M1 | Male |
| M2 | Male |
| M3 | Male |
| M4 | Male |
| M5 | Male |
| F1 | Female |
| F2 | Female |
| F3 | Female |
| F4 | Female |
| F5 | Female |

---

## List Models

### GET /v1/models

Returns a list of available models.

**Authentication:** Not required

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "supertonic-2",
      "object": "model",
      "created": 1704067200,
      "owned_by": "local"
    }
  ]
}
```

**Example:**

```bash
curl http://localhost:3000/v1/models
```

---

## List Voices

### GET /v1/voices

Returns a list of available voices.

**Authentication:** Not required

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "M1",
      "name": "M1",
      "object": "voice"
    },
    {
      "id": "F1",
      "name": "F1",
      "object": "voice"
    }
  ]
}
```

**Example:**

```bash
curl http://localhost:3000/v1/voices
```

---

## Migration from OpenAI

To migrate from OpenAI to SonicBoom:

1. Change the base URL from `https://api.openai.com` to your SonicBoom server
2. Update the endpoint from `/v1/audio/speech` to `/v1/audio/speech`
3. Keep the same authentication token (if using SonicBoom's token system)
4. Update voice names if using different voices

### Python Example

**Before (OpenAI):**

```python
from openai import OpenAI

client = OpenAI(api_key="sk-...")
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello, world!"
)
response.stream_to_file("output.mp3")
```

**After (SonicBoom):**

```python
import requests

url = "http://localhost:3000/v1/audio/speech"
headers = {
    "Authorization": "Bearer YOUR_TOKEN",
    "Content-Type": "application/json"
}
data = {
    "model": "tts-1",
    "voice": "alloy",
    "input": "Hello, world!",
    "response_format": "opus"
}

response = requests.post(url, headers=headers, json=data)
with open("output.ogg", "wb") as f:
    f.write(response.content)
```

---

## Response Formats

| Format | Content-Type | File Extension |
|--------|--------------|----------------|
| opus | `audio/ogg` | .ogg |
| mp3 | `audio/mpeg` | .mp3 |
| wav | `audio/wav` | .wav |
| aac | `audio/aac` | .aac |
| flac | `audio/flac` | .flac |

**Note:** SonicBoom internally uses Opus encoding. Other formats are transcoded.
