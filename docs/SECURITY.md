# Security

The system handles patient radiographs. This documents what is defended, how,
and what is explicitly out of scope.

## Reporting

Email <security@dentist-ai.app>. Please do not open a public issue.

---

## Authentication

| | |
|---|---|
| Hashing | Argon2id, 19 MiB / t=2 / p=1 (OWASP 2024 interactive) |
| Rehashing | Automatic on login when cost parameters are raised |
| Minimum length | 10 characters; a common-password and low-entropy check |
| Session | Signed cookie, `HttpOnly`, `SameSite=Lax`, `Secure` in production |
| Fixation | Session id regenerated on every authentication |

**User enumeration is closed.** A login attempt for an unknown address still
performs a full Argon2 verification against a dummy hash generated at startup
under the same cost parameters, and returns the byte-identical error as a
wrong password. A hard-coded dummy hash would fail to parse and return early,
restoring the timing oracle — so it is generated, not written down.

Registration does report that an email is already taken. That is a deliberate
usability trade-off, mitigated by the registration rate limit.

## CSRF

Two independent checks on every unsafe method, applied by middleware rather
than per-route so a new endpoint is protected by default:

1. `Origin` (falling back to `Referer`) must match the served host.
2. A signed token, bound to the session id, in the `X-CSRF-Token` header.

The token is **never** read from the CSRF cookie. A cross-site form post
carries cookies automatically, so accepting the cookie as proof would defeat
the mechanism entirely. The cookie exists only so the frontend can read its
own token; what actually blocks the attack is that a browser will not let a
cross-origin page set a custom header.

Covered by `tests/test_auth.py::test_csrf_cookie_alone_does_not_authorise`.

## Uploads

Uploads are the highest-risk input in the system: radiographs (JPEG, PNG,
WebP, BMP, TIFF) and 3D scans (STL, PLY, OBJ).

| Threat | Defence |
|---|---|
| Path traversal | Storage paths derive from a SHA-256 of the decoded bytes. The client filename never touches the filesystem. |
| Type confusion | Images are decoded and re-encoded through Pillow; meshes are parsed to triangles and re-emitted as binary STL. A polyglot file does not survive either path. |
| Decompression bomb | `Image.MAX_IMAGE_PIXELS` capped at ~180 MP before decode. |
| Mesh blow-up | Triangle count capped at 4 M; a declared count that does not match the file length is rejected before allocation. |
| Unbounded size | Enforced on bytes actually read, not on `Content-Length`. Separate ceilings for images (24 MB) and meshes (96 MB). |
| Metadata leakage | Re-encoding drops EXIF — which on a dental scan can carry patient name, device serial and GPS — and drops everything a mesh file carried besides geometry. |
| Blocking the loop | Decode, encode and mesh parsing run in a worker thread. |

Stored files are `0600` inside a `0700` root, outside any static mount.

## Authorisation

Every patient-bearing query filters on `organization_id`. Cross-tenant
requests return **404, not 403**: a 403 confirms the resource exists.

Images are served only by `GET /api/v1/studies/{id}/image` and meshes only by
`GET /api/v1/scans/{id}/mesh`; both verify membership and record the access.
There is no path from the public static mount to patient data.

Roles (`owner` / `dentist` / `assistant`) gate deletion, clinical review,
tooth re-charting and treatment planning.

## Transport and headers

Sent on every response:

```
Content-Security-Policy: default-src 'self'; frame-ancestors 'none';
    object-src 'none'; script-src 'self' 'sha256-…'; base-uri 'self';
    form-action 'self'; …
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: camera=(), microphone=(), geolocation=()
Cross-Origin-Opener-Policy: same-origin
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload   [production]
```

`script-src` is `'self'` plus a **hash** — not `'unsafe-inline'`. The single
inline script is the pre-paint theme setter, and its CSP hash is derived from
the source string at import time, so the two cannot drift apart. A test
asserts the hash matches the bytes actually rendered.

The application pages send `noindex, nofollow` and `referrer: same-origin`, so
study ids never leak through a `Referer` header.

## Rate limiting

| Bucket | Default | Keyed by |
|---|---|---|
| Login | 10 / 5 min | IP |
| Registration | 5 / hour | IP |
| Upload | 60 / hour | user |
| General API | 600 / min | user |

A successful login clears the login bucket, so a user who mistyped their
password a few times is not locked out afterwards.

The backend is in-process. **This is single-replica only** — behind the
`RateLimiter` protocol, so a Redis implementation is a one-line swap in the
composition root.

## Errors and logging

No exception detail reaches a client. Every error response is an RFC 9457
problem document authored in `core/errors.py`; anything unhandled becomes an
opaque 500 and the traceback goes only to the log.

Logs are structured, carry a request id, and mask email addresses
(`a***@example.com`) — enough to correlate a support ticket, not enough to
leak a user list.

## Audit trail

Every read and write of patient data writes an append-only `audit_events` row:
actor, action, resource, IP, user agent, timestamp. Covered actions include
viewing a patient, viewing a study, **accessing a study image or a 3D mesh**,
exporting, reviewing a finding, re-charting a tooth, every change to a
treatment plan, and every authentication event.

The table has no `updated_at` and no delete cascade: audit rows outlive the
records they describe.

## Secrets

`SECRET_KEY` is validated at boot — in staging or production it must be at
least 32 characters and must not be the placeholder from `.env.example`, or
the process refuses to start. Locally, an absent key generates an ephemeral
one, so sessions reset on restart rather than being signed by a well-known
constant.

`.env` is gitignored; `.env.example` documents every variable with no values.

## Out of scope

Currently **not** implemented, and would be needed for a regulated
deployment:

- Multi-factor authentication
- Encryption at rest (rely on disk/volume encryption)
- Server-side session revocation (sessions are stateless until expiry)
- Password reset flow (no email transport is wired up)
- Signed URLs for image access (access is session-based only)
- Formal HIPAA / GDPR compliance review

## Repository history

An auditor reading the git history will find an earlier version of this
application with the following issues. They are fixed in the current code and
listed so nobody has to rediscover them:

- The inference endpoint had **no authentication at all**.
- Uploaded radiographs were written into the public static directory.
- `os.path.join("uploads", file.filename)` with an attacker-controlled name.
- Exception strings were returned to the client.
- No CSRF, no rate limiting, no security headers, no session rotation.
- `SECRET_KEY` defaulted to the literal `"dev-secret"`.

**Sample radiographs and a `.env` were committed in the original history.**
Removing them from the working tree does not remove them from git. If any of
those images are real patient data, the history needs rewriting
(`git filter-repo`) and any credentials in that `.env` must be rotated.
