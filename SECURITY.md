# Security Policy

## Supported Versions

Only the latest `main` branch is supported with security fixes.

## Reporting a Vulnerability

Report vulnerabilities privately to the repository maintainers
(see [nglmercer/SonicBoom](https://github.com/nglmercer/SonicBoom)).
Do not open public issues for undisclosed vulnerabilities.

## ⚠️ Credential Rotation Notice (Operators Must Read)

Older repository revisions contained a real API credential in `tokens.json`.

**That credential must be treated as permanently compromised.** Deleting it
from Git history does not revoke it — rotation is mandatory.

If you are upgrading from an affected version:

1. Stop old instances.
2. Discard/revoke the old token store (old `tokens.json` files use a legacy
   plaintext format that the hardened server rejects outright).
3. Start the hardened server and create new tokens using the admin UI
   (`/admin`). Each new token is shown exactly once.
4. Update all API clients to the new tokens.
5. Restart services and verify old tokens no longer authenticate.

## Security Controls

SonicBoom is hardened for network-exposed deployment:

- **API tokens** are stored as SHA-256 hashes, never plaintext; raw values
  are shown once at creation and never logged or re-displayed.
- **Authentication** requires strict `Authorization: Bearer <token>` on all
  TTS, OpenAI-compatible, and queue/playback endpoints. Client-controlled
  headers (`Referer`, `Origin`, `Host`) are never trusted for auth.
- **Admin access** requires an explicitly configured strong password
  (12+ characters); the server refuses to start without one. Admin
  mutations are CSRF-protected, sessions use hardened cookies, and login
  lockout expires automatically.
- **Model supply chain**: model files are pinned to an immutable
  HuggingFace revision and verified against SHA-256 digests compiled into
  the binary before every load. Trust-on-first-use is not used.
- **Resource limits**: bounded inference concurrency, per-token rate
  limiting, request-body caps, and validated text/chunk sizes.
- **Containers** run as non-root with dropped capabilities,
  `no-new-privileges`, and digest-pinned base images.

## Deployment Guidance

- Serve production traffic over HTTPS (reverse proxy) and set
  `COOKIE_SECURE=true`. Optionally enable `ENABLE_HSTS=true` when HTTPS is
  guaranteed (see `docs/config.md`).
- Keep `tokens.json`/`0600` and `.env` out of version control.
- Set `MODEL_SHA256_JSON_PATH` only with a manifest you generated from a
  trusted download; the built-in manifest already covers the default
  revision.
- Never enable `ENABLE_SAMPLE_TOKEN` outside local development.
