"""``python -m dentist_ai`` / ``dentist-ai`` entry point."""

from __future__ import annotations

import uvicorn

from dentist_ai.core.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "dentist_ai.main:app",
        host="127.0.0.1" if settings.debug else "0.0.0.0",  # noqa: S104
        port=8000,
        reload=settings.debug,
        log_config=None,  # structlog owns logging
        proxy_headers=settings.is_production,
        forwarded_allow_ips="*" if settings.is_production else None,
    )


if __name__ == "__main__":
    main()
