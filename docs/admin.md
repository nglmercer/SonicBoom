# Admin Panel Guide

SonicBoom includes a web-based admin panel for managing API tokens and monitoring server status.

## Accessing the Admin Panel

**URL:** `http://localhost:3000/admin`

**Default Credentials:**
- Username: `admin`
- Password: `1234`

You can change these via environment variables:
```bash
export SONICBOOM_ADMIN_ID=your_username
export SONICBOOM_ADMIN_PW=your_password
```

---

## Features

### Dashboard

The main dashboard shows:
- Model loading status
- Number of active tokens
- Server health information

### Token Management

#### View Tokens

Navigate to `/admin/tokens` to see all API tokens:
- Token ID
- Creation date
- Status (active/revoked)

#### Create Token

1. Go to the tokens page
2. Click "Create New Token"
3. Copy the generated token (shown only once)
4. Share with API users

#### Revoke Token

1. Find the token in the list
2. Click "Revoke" or delete button
3. Token immediately becomes invalid

---

## Security Features

### Session Management

- Sessions are stored in-memory
- Session timeout: configurable
- Secure HTTP-only cookies

### Login Protection

The admin panel includes brute-force protection:
- **Maximum attempts:** 5 failed logins
- **Lockout duration:** 15 minutes
- IP-based tracking

After too many failed attempts, you'll see:
```
Too many login attempts. Please try again later.
```

---

## API Endpoints for Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/admin` | Admin dashboard |
| `POST` | `/admin/login` | Admin login |
| `POST` | `/admin/logout` | Admin logout |
| `GET` | `/admin/tokens` | List tokens |
| `POST` | `/admin/tokens` | Create token |
| `DELETE` | `/admin/tokens/:id` | Revoke token |

---

## Web Routes

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Home page |
| `GET` | `/health` | Health check |

---

## Best Practices

1. **Change default credentials** - Always change the default admin password
2. **Use strong tokens** - Let the system generate random tokens
3. **Rotate tokens periodically** - Revoke and recreate tokens regularly
4. **Monitor failed logins** - Check logs for suspicious activity
5. **Use HTTPS** - In production, use HTTPS to protect credentials
