# Dentist-AI

Decision support for dental clinics: a detector reads a panoramic radiograph,
the product turns its output into a per-tooth record, a report and a draft
treatment plan, and a clinician signs off on all of it.

[Quick start](#quick-start) · [Architecture](docs/ARCHITECTURE.md) · [Security](docs/SECURITY.md)

## What it does

- **Findings, not pixels.** Every detection is stored as structured data —
  class, confidence, normalised box — so the viewer filters by confidence,
  toggles classes and zooms without another request.
- **Triage, not a wall of boxes.** A crown and a caries lesion are both
  detections; only one is a problem. Findings are grouped into pathologies,
  restorations, orthodontics, anatomy and conditions, and only pathologies
  count toward "needs attention".
- **A tooth number.** Each tooth-level finding is placed on an FDI odontogram
  from its position on the image. It is an estimate; the viewer shows it as
  editable, and the clinician's correction is what gets stored.
- **A report.** Per-tooth groups, regional findings and the procedures those
  findings map to in the protocol table.
- **A treatment plan.** A draft is assembled from confirmed findings, then
  edited: steps carry a priority, a status, a tooth and an estimate in visits
  and minutes.
- **3D scans.** Intraoral scans, plaster-model scans and CBCT-derived surfaces
  in STL, PLY or OBJ, viewed in the browser with orbit, zoom and a cross
  section.
- **Patient records.** Studies, scans and plan steps in one dated timeline.
- **An audit trail.** Every read and write of patient data is recorded.

The clinician decides. Confirmations and rejections are persisted, drive the
statistics, and become the label stream for the next training run.

## Quick start

Requires Python 3.12+ and Node 22+.

```bash
make setup && make seed && make dev
```

Then open <http://127.0.0.1:8000> and sign in:

| | |
|---|---|
| email | `demo@dentist-ai.app` |
| password | `demo-clinic-2026` |

`make setup` creates the virtualenv, writes a `.env` with a generated secret
key, installs both toolchains, builds the frontend and runs migrations.

> The default inference backend is `stub` — deterministic fake detections,
> seeded from image content. It exists so the product runs end to end without
> a 2 GB torch install. See [Running the real model](#running-the-real-model).

## Commands

```bash
make help          # every target, described
make dev           # run with auto-reload
make check         # lint + typecheck + tests (what CI runs)
make test-cov      # tests with an HTML coverage report
make watch         # rebuild the frontend on change
make migration m="add something"
make docker-up     # app + Postgres, production-shaped
```

## Running the real model

```bash
pip install -e ".[ml]"          # torch + ultralytics
cp /path/to/best.pt models/best.pt
```

Then in `.env`:

```ini
DENTIST_AI__ML__BACKEND=yolo
DENTIST_AI__ML__WEIGHTS_PATH=models/best.pt
DENTIST_AI__ML__DEVICE=cpu      # or mps, cuda, 0
```

Weights are not committed. They are large, opaque to review, and
version-controlling them makes every clone pay for every past revision.

To train, see [`training/`](training/):

```bash
python training/train.py --epochs 100 --device mps
```

The trainer verifies the dataset's class list against
`dentist_ai.ml.taxonomy` before starting, because a silently reordered class
list produces a model that mislabels every finding.

## Architecture

```
src/dentist_ai/
├── core/        config, security, logging, errors, rate limiting, ids
├── db/          SQLAlchemy models and session management
├── ml/          detector protocol, YOLO backend, stub backend, taxonomy
├── clinical/    FDI charting, treatment protocols, report assembly, labels
├── schemas/     Pydantic request/response contracts
├── services/    business logic — auth, patients, studies, scans, meshes,
│                treatment, storage, audit
├── api/         HTTP layer — routers, dependencies, middleware, presenters
├── web/         server-rendered pages and Vite asset resolution
├── templates/   Jinja2
└── static/      images, icons, built frontend bundle

frontend/src/
├── styles/      design tokens, base, components, per-area styles
├── lib/         typed API client, DOM helpers, forms, toasts, theme
├── features/    one module per screen, loaded on demand
└── entries/     marketing · auth · app
```

Layers depend downward only: `api → services → db`. Nothing in `services`
imports FastAPI, and nothing in `core` imports the application.

`clinical/` is separate from `ml/`: tooth numbers, procedures and
report text are rules a clinician can read, not model output.

Full rationale, including trade-offs and known scaling limits, is in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Tech

| | |
|---|---|
| Backend | FastAPI, SQLAlchemy 2.0 (async), Alembic, Pydantic v2 |
| Database | PostgreSQL in production, SQLite for local development |
| Inference | Ultralytics YOLO behind a swappable protocol |
| 3D | STL/PLY/OBJ parsed with NumPy, rendered with hand-written WebGL |
| Frontend | Vite, TypeScript (strict), hand-written CSS with design tokens |
| Quality | ruff, mypy (strict), pytest, GitHub Actions |

No frontend framework and no 3D library. The product is server-rendered pages
plus small per-screen TypeScript modules; the mesh viewer is one WebGL program
over one buffer, which is the whole requirement for opaque triangles under a
fixed light.

## Configuration

All settings live in `.env`, validated at boot by `core/config.py`. Nested
values use a double underscore:

```ini
DENTIST_AI__DATABASE__URL=postgresql+asyncpg://user:pass@host/db
DENTIST_AI__ML__BACKEND=yolo
DENTIST_AI__SECURITY__LOGIN_RATE_LIMIT=10/5m
DENTIST_AI__STORAGE__MAX_MESH_BYTES=100663296
```

See [`.env.example`](.env.example) for the full list. A misconfigured
deployment fails at startup with a precise message rather than at the first
request.

## Testing

```bash
make test
```

The suite drives the real HTTP stack — middleware, CSRF, cookies, database —
through an in-process ASGI transport. Only the detector is swapped, and even
that is a first-class backend rather than a mock.

Covered: tenant isolation across clinics, CSRF (including that the cookie
alone grants nothing), login-failure indistinguishability, path traversal in
uploads, EXIF stripping, decompression-bomb limits, mesh decoding across all
five supported encodings, FDI numbering geometry, protocol-table integrity,
and that every page actually links a stylesheet.

## Deployment

```bash
docker compose up --build
```

The image runs as an unprivileged user, migrations run to completion before
the app starts, and patient storage is a mounted volume. Behind a reverse
proxy, terminate TLS there and keep `DENTIST_AI__ENVIRONMENT=production` so
cookies are `Secure` and HSTS is sent.

## Scope

Dentist-AI is a decision-support tool. It does not diagnose, and it does not
replace a clinician's reading of a radiograph. Tooth numbers are geometric
estimates and treatment steps are lookups in a protocol table — neither is a
clinical judgement. Every screen that shows a finding says so.

## Licence

Proprietary. All rights reserved.
