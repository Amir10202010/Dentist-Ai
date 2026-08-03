# Dentist-AI

Decision support for dental clinics. A detector reads a panoramic radiograph
and a staged pipeline reads a CBCT volume; the product turns both into a
per-tooth record, a report and a draft treatment plan, and a clinician signs
off on all of it.

[Quick start](#quick-start) · [Architecture](docs/ARCHITECTURE.md) · [Deployment](docs/DEPLOY.md) · [Security](docs/SECURITY.md)

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
- **A treatment plan.** Confirmed findings across radiographs and CBCT are
  turned into options of differing scope, each with its own sequence, visit
  count, chair time, duration and complexity, and with risks derived from the
  combination of findings and procedures rather than listed per finding.
  Nothing is scheduled until a clinician accepts an option; steps then carry a
  priority, a status, a tooth and an estimate in visits and minutes.
- **CBCT, read in depth.** A DICOM series or a NIfTI volume is decoded to a
  canonical container and put through five stages — quality, segmentation,
  detection, classification, synthesis — producing findings across 20 classes,
  each with a confidence, an affected region, a measurement in millimetres,
  the reason it was called, and what to do next.
- **A viewer that measures.** Three synchronised planes with a shared
  crosshair, plus a volumetric 3D render; distances and angles in millimetres,
  point density in HU, annotations, window presets, per-axis clipping, and
  findings drawn over both the slices and the volume.
- **A case assistant.** Answers questions about one case from stored rows
  only — why a finding was called, what the options are, what to check at the
  appointment — and cites the records each answer was built from.
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
│   ├── stages/  quality · segmentation · detection · classification · synthesis
│   ├── pipeline.py   stage protocol, registry, per-stage failure isolation
│   ├── volumetrics.py NumPy image analysis — thresholding, components, metrics
│   └── cbct_taxonomy.py the 20 volume finding classes and what each implies
├── clinical/    FDI charting, treatment protocols, report assembly, labels,
│                treatment planner
├── schemas/     Pydantic request/response contracts
├── services/    business logic — auth, patients, studies, scans, meshes,
│                volumes, planning, assistant, search, library, collaboration,
│                timeline, notifications, analytics, treatment, storage, audit
├── api/         HTTP layer — routers, dependencies, middleware, presenters
├── web/         server-rendered pages and Vite asset resolution
├── templates/   Jinja2
└── static/      images, icons, built frontend bundle

frontend/src/
├── styles/      design tokens, base, components, per-area styles
├── lib/         typed API client, DOM helpers, forms, toasts, theme, DVOL
├── features/    one module per screen, loaded on demand — including the
│                volume viewer: MPR, WebGL2 renderer, shared store
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
| Inference | Ultralytics YOLO behind a swappable protocol; the CBCT pipeline is NumPy only |
| Imaging | DICOM Part 10 and NIfTI-1 decoded without an imaging dependency |
| 3D | Volume ray marching on WebGL2 3D textures; STL/PLY/OBJ meshes parsed with NumPy |
| Frontend | Vite, TypeScript (strict), hand-written CSS with design tokens |
| Quality | ruff, mypy (strict), pytest, GitHub Actions |

No frontend framework and no 3D library. The product is server-rendered pages
plus small per-screen TypeScript modules; the mesh viewer is one WebGL program
over one buffer, and the volume renderer is a second one that marches rays
through a single 3D texture. Reslicing for the 2D planes is done on the CPU,
because a plane is a memory-access pattern rather than a shading problem.

The CBCT pipeline carries no learned weights. Findings come from thresholding,
connected components and geometric rules over the volume, so every call can be
explained in terms a clinician can check — which is also why each one ships
with its rationale rather than a score alone.

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

The CBCT half is tested against generated phantoms rather than patient data.
[`scripts/synthetic_cbct.py`](scripts/synthetic_cbct.py) builds anatomically
arranged volumes for seven presets — healthy, periapical, cyst, implant site,
restored, periodontal, poor quality — and the suite asserts both that findings
appear where the phantom put them and, on the healthy preset, that anatomy is
not reported as pathology. That second direction is the one that matters: a
classifier which flags a normal mandibular canal as a lesion is worse than one
that misses it.

CI additionally runs the migrations against a real PostgreSQL — upgrade,
downgrade to base, upgrade again — because SQLite accepts schema mistakes that
Postgres rejects.

## Deployment

```bash
docker compose up --build
```

The image runs as an unprivileged user, migrations run to completion before
the app starts, and patient storage is a mounted volume. Behind a reverse
proxy, terminate TLS there and keep `DENTIST_AI__ENVIRONMENT=production` so
cookies are `Secure` and HSTS is sent.

[`render.yaml`](render.yaml) is a ready Blueprint — Docker service, managed
Postgres, a disk for patient volumes, a generated signing key — but nothing is
Render-specific beyond that file. Serverless hosts are not an option: volumes
live on disk, analysis outruns a function timeout, and pages are rendered by
the process that serves the JSON. [docs/DEPLOY.md](docs/DEPLOY.md) covers the
reasoning, the required environment variables, and other container hosts.

## Scope

Dentist-AI is a decision-support tool. It does not diagnose, and it does not
replace a clinician's reading of a radiograph. Tooth numbers are geometric
estimates and treatment steps are lookups in a protocol table — neither is a
clinical judgement. Every screen that shows a finding says so.

## Licence

Proprietary. All rights reserved.
