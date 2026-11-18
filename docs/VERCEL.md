# Public demo on Vercel

Live: <https://dentist-ai-zeta.vercel.app>

This is a **demo**, not the production setup (that is still `render.yaml` / Docker).

## How it works

- `api/index.py` is the only function. On a cold start it copies a pre-seeded
  SQLite database and its scan storage from `vercel/seed/` into `/tmp`, then
  serves the normal FastAPI app with the `stub` detector.
- Anything a visitor changes lives only in that function instance. The next
  cold start is a clean demo clinic again.
- CSS, JS and images are served from Vercel's CDN out of `public/static`. They
  cannot come from the function: Vercel's Python bridge resets the
  content-type of function responses to `text/html`, and with `nosniff` the
  browser refuses such stylesheets.
- The only secret is `DENTIST_AI__SECRET_KEY`, set in the Vercel project's
  production environment.

Demo login: `demo@dentist-ai.app` / `demo-clinic-2026` (from `scripts/seed.py`).

## Deploy

```bash
./vercel/deploy.sh            # production
./vercel/deploy.sh --preview  # preview URL
```

`public/` is built by the script and is not committed.

## Re-seeding the demo data

```bash
rm -rf vercel/seed && mkdir -p vercel/seed/storage
export DENTIST_AI__DATABASE__URL="sqlite+aiosqlite:///$PWD/vercel/seed/dentist_ai.sqlite3"
export DENTIST_AI__STORAGE__ROOT="$PWD/vercel/seed/storage"
export DENTIST_AI__SECRET_KEY=seed-only DENTIST_AI__ML__BACKEND=stub
alembic upgrade head && python scripts/seed.py
```
