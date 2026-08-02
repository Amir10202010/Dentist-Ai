# Dentist-AI — developer entry points.
#
# `make setup` then `make dev` is the whole onboarding story.

SHELL := /bin/bash
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
export PATH := $(CURDIR)/$(VENV)/bin:$(PATH)

.DEFAULT_GOAL := help
.PHONY: help setup install-frontend dev serve build assets migrate migration \
        seed test test-cov lint fmt typecheck check clean docker-build docker-up \
        docker-down reset-db

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup: ## Create the venv, install everything, build assets, migrate
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip --quiet
	$(PIP) install -e ".[dev]" --quiet
	@test -f .env || (cp .env.example .env && \
		$(PY) -c "import pathlib,secrets; p=pathlib.Path('.env'); \
p.write_text(p.read_text().replace('change-me-generate-a-real-one-before-running-anywhere', secrets.token_urlsafe(48)))" && \
		echo "→ wrote .env with a generated SECRET_KEY")
	@mkdir -p var models
	$(MAKE) install-frontend build migrate
	@echo ""
	@echo "✓ Ready. Run 'make seed' for demo data, then 'make dev'."

install-frontend: ## Install frontend dependencies
	cd frontend && npm install --silent

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
dev: ## Run the app with auto-reload (http://127.0.0.1:8000)
	$(VENV)/bin/uvicorn dentist_ai.main:app --reload --host 127.0.0.1 --port 8000

serve: ## Run the app as production would (no reload)
	$(VENV)/bin/uvicorn dentist_ai.main:app --host 0.0.0.0 --port 8000 --workers 2 \
		--proxy-headers --forwarded-allow-ips '*'

build: ## Build the frontend bundle
	cd frontend && npm run build

watch: ## Rebuild the frontend on change
	cd frontend && npm run dev

assets: ## Regenerate brand images and icons
	$(PY) scripts/generate_assets.py

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
migrate: ## Apply migrations
	$(VENV)/bin/alembic upgrade head

migration: ## Autogenerate a migration: make migration m="add x"
	@test -n "$(m)" || (echo "usage: make migration m=\"description\"" && exit 1)
	$(VENV)/bin/alembic revision --autogenerate -m "$(m)"
	$(VENV)/bin/ruff format migrations/versions/

downgrade: ## Roll back one migration
	$(VENV)/bin/alembic downgrade -1

seed: ## Load a demo clinic with patients and analysed studies
	$(PY) scripts/seed.py

reset-db: ## Drop the local database and re-migrate (destructive)
	rm -f var/dentist_ai.sqlite3*
	$(MAKE) migrate

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
test: ## Run the test suite
	$(PY) -m pytest

test-cov: ## Run tests with a coverage report
	$(PY) -m pytest --cov --cov-report=term-missing --cov-report=html
	@echo "→ htmlcov/index.html"

lint: ## Lint Python and TypeScript
	$(VENV)/bin/ruff check .
	cd frontend && npm run typecheck

fmt: ## Format and autofix
	$(VENV)/bin/ruff check . --fix
	$(VENV)/bin/ruff format .

typecheck: ## Run mypy (strict)
	$(VENV)/bin/mypy

check: lint typecheck test ## Everything CI runs

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-build: ## Build the container image
	docker build -t dentist-ai:latest .

docker-up: ## Start the full stack (app + Postgres)
	docker compose up --build

docker-down: ## Stop the stack
	docker compose down

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
clean: ## Remove build and cache artefacts
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage
	rm -rf src/dentist_ai/static/dist
