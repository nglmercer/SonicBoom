# Admin Panel Guide

SonicBoom includes a web-based admin panel for managing API tokens and monitoring server status.

## Accessing the Admin Panel

**URL:** `http://localhost:3000/admin`

**Credentials:** configured via environment variables — there are no defaults:

```bash
export SONICBOOM_ADMIN_ID=admin
export SONICBOOM_ADMIN_PW=a-strong-password-with-12-plus-chars
```

The server refuses to start when `SONICBOOM_ADMIN_PW` is missing, shorter
than 12 characters, or a well-known default (`1234`, `password`, `admin`,
...). Generate one, e.g.:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(24))"
```

---

## Features

### Dashboard

The main page shows the token list (ID, fingerprint, creation/expiry, status)
and the token creation form.

> Raw bearer tokens are **never** displayed in the token list — only a
> non-sensitive fingerprint (first 8 hash chars). A new token is shown
> **exactly once** right after creation:
>
> `Copy this token now. It cannot be displayed again.`

### Token Management

#### View Tokens

Navigate to `/admin` to see all API tokens:

- Token ID
- Fingerprint (e.g. `abcd1234…`, not usable for auth)
- Creation date
- Expiration date
- Status (active/expired/revoked)

#### Create Token

1. Go to `/admin`
2. Pick an optional expiry date
3. Click "Generate Token"
4. Copy the generated token immediately (shown only once)
5. Share with API users over a secure channel

Tokens are stored as SHA-256 hashes; the raw value cannot be recovered later.

#### Revoke Token

1. Find the token in the list
2. Click "Revoke"
3. Token immediately becomes invalid

#### Logout

Logout is a `POST /admin/logout` form button (CSRF-protected), not a link.

---

## Security Features

### Session Management

- Sessions are stored server-side (in-memory)
- Session id is rotated on every successful login
- Session is destroyed server-side on logout
- Cookie: `HttpOnly`, `SameSite=Strict`, `Secure` when `COOKIE_SECURE=true`,
  8-hour inactivity expiry by default (`ADMIN_SESSION_EXPIRY_SECS`)
- Serve production deployments over HTTPS and set `COOKIE_SECURE=true`

### CSRF Protection

All admin mutations (`POST /admin/tokens`,
`POST /admin/tokens/{id}/revoke`, `POST /admin/logout`) require a
per-session CSRF token sent as a hidden form field. Requests with a missing
or invalid token are rejected with `403`.

### Login Protection

The admin panel includes brute-force protection with automatic expiry:

- **Window:** 5 failed logins within 10 minutes
- **Lockout:** 15 minutes, then automatic recovery
- IP-based tracking (socket peer by default; forwarded headers are only
  honored from explicitly configured trusted proxies — see
  `TRUST_PROXY`/`TRUSTED_PROXIES` in [config.md](config.md))
- Bounded tracker memory (oldest entries evicted under pressure)

After too many failed attempts, you'll see:

```
Too many failed attempts. Try again later.
```

with HTTP status `429`.

---

## API Endpoints for Admin

| Method | Endpoint                    | Description      |
|--------|-----------------------------|------------------|
| `GET`  | `/admin`                    | Admin dashboard  |
| `GET`  | `/admin/login`              | Login page       |
| `POST` | `/admin/login`              | Admin login      |
| `POST` | `/admin/logout`             | Admin logout     |
| `POST` | `/admin/tokens`             | Create token     |
| `POST` | `/admin/tokens/{id}/revoke` | Revoke token     |

---

## Web Routes

| Method | Endpoint  | Description                      |
|--------|-----------|----------------------------------|
| `GET`  | `/`       | Home page (TTS demo, needs token)|
| `GET`  | `/health` | Liveness probe (always 200)      |
| `GET`  | `/ready`  | Readiness probe (model loaded)   |

---

## Best Practices

1. **Use a strong admin password** — the server enforces 12+ characters
2. **Use strong tokens** - Let the system generate random tokens
3. **Rotate tokens periodically** - Revoke and recreate tokens regularly
4. **Monitor failed logins** - Check logs for suspicious activity
5. **Use HTTPS** - In production, terminate TLS (reverse proxy) and set `COOKIE_SECURE=true`
6. **Restrict file permissions** - keep `tokens.json`/`0600` and `.env` out of Git
