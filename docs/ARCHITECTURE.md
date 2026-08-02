# Architecture

Why the system is shaped the way it is, and where it will need to change.

## Shape

```
Browser
  │  server-rendered HTML + per-screen TS module
  ▼
FastAPI  ── middleware: request id → gzip → security headers → CSRF
  │
  ├── web/      Jinja pages
  └── api/v1/   JSON, OpenAPI-documented
        │  dependencies inject services
        ▼
      services/   business logic, no HTTP types
        │
        ├── db/        SQLAlchemy async
        ├── ml/        Detector protocol (2D) · Stage pipeline (3D)
        ├── clinical/  tooth numbering, protocols, report assembly
        └── storage    content-addressed files on disk
```

Dependencies point one way. `services` never imports FastAPI; `core` never
imports the application. That is what makes the services testable without a
client and the detector swappable without touching a route.

## Decisions

### Server-rendered pages, not a SPA

The product is a handful of screens, mostly forms plus two viewers. A
client-side framework would add a router that must mirror the server's, a
hydration step, and ~40 KB of runtime before a single feature.

Instead: Jinja renders the page, a small TypeScript module hydrates the parts
that need behaviour, and Rollup splits one module per screen. The dashboard
never downloads the WebGL mesh viewer.

The cost is real: no shared client state between screens, and every navigation
is a round-trip. At this size that is a good trade. If the product grows a
multi-pane workspace where state must survive navigation, revisit it.

### Detections stored as data, not baked into images

Findings are rows: class, confidence, and a box normalised to `[0, 1]`. That
is what makes confidence filtering, class toggling, severity triage, the
odontogram, CSV export and clinician review possible, and it means toggling a
class costs no network.

Coordinates are normalised rather than pixel-based so re-encoding or
downscaling the master image can never invalidate stored findings.

### The detector is a protocol

`ml/detector.py` defines `Detector`; `yolo.py` and `stub.py` implement it.

The stub is not a test mock — it is the default backend. It produces
content-seeded pseudo-detections so a fresh clone runs the whole product
without a 2 GB install, and so CI exercises every HTTP path in milliseconds.
The YOLO backend defers its `ultralytics` import to first load, so a stub
deployment never pays for torch.

Inference runs in a bounded thread pool with a semaphore, because
`model.predict` is a multi-second CPU-bound call and the model object is not
thread-safe.

### Clinical inference is rules, model inference is the model

`clinical/` holds three things the model does not produce:

- **`charting.py`** turns a box position into an FDI tooth number. It is
  geometry over two assumptions about panoramic layout — a midline at `x=0.5`
  and an occlusal plane that sags toward it — so the number is an estimate.
  The UI presents it as editable and stores the clinician's correction with
  `tooth_confirmed`, which is the value that counts as fact.
- **`protocols.py`** is an explicit `finding class → procedure` table. A plan
  item never comes from a model; it comes from a row here that a clinician
  accepted.
- **`report.py`** assembles counts, per-tooth groups and protocol lookups into
  the study report. Nothing in it is generated prose.

Keeping this out of `ml/` is the point: it is the part of the product a
clinician can audit line by line.

### Content-addressed private storage

Files are keyed by the SHA-256 of their *decoded, re-encoded* bytes, under
`var/storage/…`, outside any static mount. Radiographs become one normalised
JPEG; 3D scans become one binary STL whatever they arrived as.

That single rule pays for several things: identical uploads deduplicate, the
client's filename never touches the filesystem, and re-encoding destroys
polyglot files and strips EXIF, which on a dental scan can carry the patient's
name.

The only routes out are `GET /api/v1/studies/{id}/image` and
`GET /api/v1/scans/{id}/mesh`, which check organisation membership and write
an audit row.

### 3D scans are normalised on ingest, not on read

Scanners export STL, PLY and OBJ, in ASCII and binary, with colour, normals
and texture coordinates. `services/mesh.py` decodes all of it and re-emits
binary STL.

The browser therefore needs one parser instead of four (about thirty lines in
`features/mesh-viewer.ts`), a file that claims to be a mesh but is not gets
rejected at the door, and the content hash is a function of the geometry
rather than of whichever exporter wrote the file.

Surface scans get no automatic analysis. The mesh viewer is an archive and a
renderer; claiming otherwise would be claiming a detector that does not exist.
Volumetric scans are a different matter — see below.

### CBCT is a pipeline, not a detector

A radiograph is one image in, boxes out, and `ml/detector.py` is the right
shape for it. A CBCT reconstruction is not. Deciding whether a lucency at a
root apex is a lesion needs the volume segmented first; deciding whether the
*finding* is trustworthy needs to know whether the patient moved. Those are
separate questions that fail separately and will be replaced by trained
networks on separate schedules.

So `ml/pipeline.py` is a small orchestrator over a `Stage` protocol, and
`ml/cbct.py` is the only module that knows the order:

```
quality control → segmentation → detection → classification → synthesis → treatment
```

Three properties are worth the indirection:

- **Failures are isolated.** A stage that throws marks itself failed and the
  run continues. One exception discarding an eight-second analysis is the wrong
  trade for clinical software, so `RunRecord.succeeded` asks only whether the
  *blocking* stages survived.
- **Detection and classification are separate.** Detection answers "something
  is here and here is what it measures"; classification answers "this is what
  it is". Either can be replaced independently, and a detector tuned for
  sensitivity can be paired with a classifier tuned for specificity.
- **Every run is described.** `ai_runs` stores the per-stage log. "The AI found
  a cyst" is not reviewable; "the detection stage, v3, found it in 240 ms on a
  volume the QC stage scored 0.82" is.

### The classifier has gates, not just weights

`ml/stages/classification.py` gives every finding class two parts: a **gate** of
conditions that are definitional, and a **score** of conditions that are
evidential. An implant is buried in bone below the crest — a dense body at the
occlusal plane with air above it is a crown, and no amount of being the right
size makes it a fixture.

Folding both into one weighted average is the obvious way to write this and it
is where heuristic classifiers go wrong: enough weak positive evidence outvotes
a hard anatomical impossibility. Keeping them apart also produces a better
explanation, which is what the finding panel and the report read back.

Density carries more weight than shape, because it separates the pairs geometry
cannot. A cyst and a marrow space are the same size and shape; one is fluid and
the other is bone. A lesion and a maxillary sinus are both radiolucent; one is
water density and the other is air.

**What the default backend is.** There is no trained volumetric network here.
The stages are real image analysis — Otsu thresholding, connected components,
morphological enclosure tests, crest-height profiling — applied by explicit
clinical rules. Findings are derived from the voxels rather than seeded from a
hash, which is what makes the viewer, the report and the finding list honest
end to end. What it costs is sensitivity on exactly what a convolutional
network is best at: early caries, subtle resorption, a hairline fracture. The
confidences are capped accordingly, and the two classes that can only ever be a
referral are capped hardest.

### CBCT volumes are normalised to one 8-bit container

`services/volume.py` decodes DICOM series (explicit and implicit VR, both byte
orders), multi-frame DICOM and NIfTI-1, and re-emits `DVOL`: a fixed 64-byte
header followed by 8-bit voxels. Same reasoning as binary STL — one client
parser, rejection at the door, a content hash over the voxels rather than over
whichever console wrote the study.

Two numbers in it are load-bearing:

**Eight bits, not sixteen.** A reconstruction carries ~12 bits of real signal,
but the viewer's job is windowing, and a window maps a range onto 256 display
levels however many bits went in. Eight halves the transfer and lets WebGL
sample the volume as `R8` with no conversion. The linear map back to Hounsfield
units rides in the header, so a voxel readout still reports HU.

**Decimated on ingest.** A 0.2 mm full-arch scan is 600³ voxels — more than a
browser will hold as a 3D texture. Ingest block-averages by an integer factor
per axis until every axis fits `max_volume_dimension`, and **scales the stored
spacing to match**. That second half is the part that matters: if it were
missed, every measurement in the viewer would be wrong by the decimation
factor, which is the one way this product could produce a confidently incorrect
number. `tests/test_volume_codec.py` asserts the field of view in millimetres
survives decimation for that reason.

Compressed transfer syntaxes (JPEG, JPEG 2000, RLE) are refused by name rather
than decoded, and the refusal survives an archive in which *every* instance is
compressed — the common real-world failure, where a generic "no readable
slices" would hide the one fact that tells the clinic how to fix its export.

### A plan is a choice, not an answer

`clinical/treatment_planner.py` is the third explicit table in `clinical/`,
beside `charting` and `protocols`, and it is there for the same reason: a plan
proposes doing something to a person, so it has to be auditable line by line.
Nothing in it is model output.

It returns **three options** — conservative, standard, comprehensive — rather
than one recommendation, because the same findings support several defensible
courses and which is right depends on what the patient wants and can afford.
They are not tiers: the conservative option is the correct answer for someone
who wants the problem treated and nothing more, and the interface renders them
as equal cards for that reason. Options that collapse into a narrower one (a
case with nothing elective in it) are dropped rather than shown twice.

Two details carry most of the value:

- **Sequencing is a constraint chain, not a sort order.** Disease is controlled
  before anything is restored, extraction sites heal before implants go into
  them, and a diagnostic referral precedes the surgery it informs.
- **Risk comes from the combination.** An extraction is routine; an extraction
  in a mandible whose canal runs close to the roots is a nerve-injury
  conversation. Neither finding produces that warning alone, which is why
  `_RISK_RULES` keys on pairs. A warning that fires on every extraction is a
  warning that gets ignored.

Durations are quoted in **calendar weeks including healing**, not chair time.
An implant case is two hours of work over four months, and quoting the two
hours is how a patient ends up feeling misled.

Generating writes a **draft** with options and no items. `services/planning.py`
turns one option's steps into plan items only when a clinician accepts it, and
re-accepting is refused rather than silently replacing a schedule they may
already have edited.

### The assistant answers from the record, and cannot do otherwise

`services/assistant.py` is a retrieval and composition engine over one case,
not a language model. A question is matched to an intent, the intent's handler
reads the relevant rows, and the answer is assembled from them. It cannot
produce a claim that is not already stored, and every answer carries citations
naming the rows it used — which is the property that makes it usable on patient
data at all.

The coverage limit is deliberate and visible: an unrecognised question gets the
list of what *is* answerable rather than a guess, and the UI styles that
differently from an answer.

One implementation note that is easy to get wrong: intent matching is on **token
prefixes**, not substrings. A plain `in` test routes "сколько стоит имплант" to
the treatment handler, because *имплант* contains *план* — a wrong answer that
is invisible in review and obvious to a user.

Where a real language model belongs here is as a *rephraser* of answers this
module has already grounded, never as their source. The seam is
`AssistantService.answer`, which returns structured text plus citations rather
than a finished string.

### The viewer is four renderers over one state

`features/volume-state.ts` holds a single store; the three MPR panes and the
volume renderer subscribe to it. Clicking a lesion in the axial pane moves the
other two slices, swings the 3D clipping plane and updates the readout because
they all read the same object — not because four update calls are kept in step
by hand.

The reslices are drawn on the **CPU**: a plane is one strided read out of an
array the browser already holds, windowed into an `ImageData`. Uploading a 3D
texture to sample three axis-aligned slices out of it would add a GPU
dependency to the half of the viewer that has to work everywhere. Each slice is
rendered at native voxel resolution and blitted through a transform carrying
the anisotropic aspect correction — not cosmetic, since a coronal reslice drawn
at one pixel per voxel on a 0.3 × 0.3 × 0.6 mm scan is half its true height and
every measurement taken off it is wrong.

The volume rendering is **WebGL2 only**, because `sampler3D` and `R8` textures
do not exist in WebGL1 and the alternatives cost more code and look worse.
Where it is unavailable the pane says so and the three MPR panes — the
diagnostically important half — carry on.

### Multi-tenancy by explicit scoping

Every patient-bearing table carries `organization_id`, every query filters on
it, and composite indexes lead with it. Cross-tenant lookups return **404, not
403** — a 403 would confirm that another clinic's record exists.

This is enforced by convention plus tests (`tests/test_tenancy.py`,
`tests/test_scans.py`) rather than by row-level security. RLS in Postgres
would be stronger; it was not adopted because it does not work on SQLite and
would split local development from production. If the tenant count grows
beyond a few hundred, revisit.

### Sessions, not JWTs

Signed cookies carrying an opaque session id. A JWT would move state to the
client and make revocation an unsolved problem, in exchange for statelessness
this app does not need.

The session id is regenerated on every login, which closes session fixation.
CSRF is a signed token bound to that session id, required in a custom header —
never accepted from the CSRF cookie alone, because a cross-site form post
would carry that cookie automatically.

### Inference runs inline

Upload → analyse → respond, in one request. On CPU that is a couple of
seconds, and a synchronous result is a better experience than an upload that
returns "pending" and a poll loop.

The `Study.status` column already models `pending/processing/completed/failed`,
so moving to a queue is a routing change rather than a migration. The trigger
to do so is either a model slow enough that requests time out, or a need to
retry failures automatically.

## Data model

| Table | Purpose |
|---|---|
| `organizations` | Tenant. Every other table hangs off this. |
| `users` | Members, with a coarse role (owner / dentist / assistant). |
| `patients` | Chart records. Soft-deleted via `archived_at`. |
| `studies` | One radiograph plus its analysis run. |
| `findings` | One detection: class, confidence, normalised box, tooth number, review state. |
| `scans_3d` | One 3D scan: source format, triangle count, bounds, arch. |
| `treatment_plans` | An ordered set of proposed and agreed work for a patient. |
| `treatment_plan_items` | One step: procedure, tooth, priority, status, estimate. |
| `volumes` | One CBCT reconstruction plus its analysis run. |
| `volume_findings` | One volumetric detection: class, confidence, normalised prism, region, frozen rationale. |
| `measurements` | A distance, angle or density probe a clinician took in the viewer. |
| `annotations` | A note pinned to a point in a volume or on a radiograph. |
| `ai_runs` | Per-stage log of one pipeline execution. Outlives the resource. |
| `treatment_options` | Alternative approaches to the same case, side by side. |
| `patient_notes`, `appointments` | Dated entries on the patient timeline. |
| `comments`, `review_assignments` | Collaboration, scoped by `resource_type`/`resource_id`. |
| `notifications` | Per-user notification centre, fanned out at write time. |
| `case_entries` | Teaching cases: a copy, not a view over the live record. |
| `assistant_threads`, `assistant_messages` | Case-grounded Q&A, with citations. |
| `audit_events` | Append-only access trail. Outlives what it describes. |

Details worth knowing:

**`patients.search_text`** is a denormalised, Python-lower-cased concatenation
of name, phone, chart number and email. SQL `lower()` folds ASCII only on
SQLite, so "Иванов" never matches "иванов" there while working correctly on
Postgres. Folding in Python gives one behaviour on both, and collapses three
OR'd predicates into one indexed column.

**Primary keys use `BigInteger().with_variant(Integer, "sqlite")`.** SQLite
only autoincrements a column declared exactly `INTEGER PRIMARY KEY`; a plain
`BIGINT` fails every insert. The variant keeps Postgres on 64-bit ids.

**Plan items copy their estimate from the protocol table** instead of reading
it back. An agreed plan should not be silently re-priced when the table gains
an entry.

**`volume_findings` freeze their rationale and next steps** at analysis time
rather than looking them up on read, so a report printed today still reads the
way it did when a clinician signed it. `requires_confirmation` is the exception
and *is* read live from the taxonomy: it is a policy about how a class may be
presented, and a stored copy would let a finding written before the policy
tightened escape it.

**A re-analysis preserves the clinician's adjudication.** Findings are matched
across runs by class and position to two decimal places — about a millimetre on
a dental field of view — so a confirmation and a hand-corrected tooth number
survive a pipeline upgrade while the model's own opinion is free to change.

**`comments` and `review_assignments` key on `resource_type`/`resource_id`**
rather than a nullable foreign key per kind, whose alternative is a column and
a CHECK constraint added every time the product gains a screen.

**Case sharing is to a *user*, never a link.** A URL that grants access to a
patient record without a session is a liability this product does not need, so
`review_assignments` names a colleague and the audit trail keeps naming a
person.

## Known limits

| Limit | When it bites | Fix |
|---|---|---|
| Patient search is `LIKE '%x%'` | Tens of thousands of patients per clinic | `pg_trgm` GIN index on `search_text` |
| Rate limiter is in-process | More than one replica | Redis backend behind the existing `RateLimiter` protocol |
| Storage is a local filesystem | Multiple app nodes | S3-compatible object store behind `ImageStorage` / `MeshStorage` |
| Inference is inline | Model slower than the request timeout | Queue, using the existing `Study.status` states |
| Sessions are stateless cookies | Need to revoke a specific session | Server-side session table keyed by `session_id` |
| Meshes are parsed in the request | Very large scans on a busy box | Parse in a worker, keyed off a `pending` state like studies |
| CBCT analysis runs in the request | A field of view large enough to pass the request timeout | Queue, using the existing `Volume.status` states |
| Volumes are decimated to 256³ | A clinician needs sub-millimetre detail in the viewer | Serve a full-resolution region of interest on demand |
| The CBCT classifier is rules, not a network | Sensitivity on early caries, resorption, hairline fractures | A trained stage implementing the same `Stage` protocol |
| Crowns in contact cannot be separated by thresholding | Per-tooth findings in a fully dentate posterior segment | Tooth instance segmentation |
| Tooth numbering assumes a well-positioned OPG | Tilted or rotated captures | Clinician correction, which is already the stored value |
| The assistant answers a fixed intent vocabulary | A question outside it | A grounded rephrasing layer over `AssistantService.answer` |
| Plan durations are practice-independent estimates | A clinic whose scheduling differs | Per-item edits, which already exist |

Each of these sits behind an interface already.

## Testing strategy

Tests run against the real application through an in-process ASGI transport:
real middleware, real CSRF, real cookies, real database. Only the detector is
substituted, and it is a shipped backend rather than a mock.

The bias is toward tests that would catch a security or correctness
regression, not toward line coverage. `tests/test_assets.py` exists because a
build-plumbing bug once shipped an entirely unstyled site while every other
test passed.
