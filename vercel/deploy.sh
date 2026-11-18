#!/usr/bin/env bash
# Build the frontend, stage static files for Vercel's CDN, and deploy the demo.
#
# Vercel serves `public/` only when it is uploaded ready-made, and its Python
# bridge resets the content-type of function responses, so CSS/JS must come
# from the CDN rather than from FastAPI's StaticFiles.
#
#   ./vercel/deploy.sh          # production
#   ./vercel/deploy.sh --preview
set -euo pipefail
cd "$(dirname "$0")/.."

(cd frontend && npm ci --no-audit --no-fund && npm run build)
rm -rf public && mkdir -p public && cp -R src/dentist_ai/static public/static

if [[ "${1:-}" == "--preview" ]]; then
  npx --yes vercel@latest deploy --yes
else
  npx --yes vercel@latest deploy --prod --yes
fi
