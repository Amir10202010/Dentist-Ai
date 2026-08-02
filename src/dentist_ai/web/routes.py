"""Server-rendered pages.

The application shell is rendered on the server and hydrated by a small
per-screen TypeScript bundle. There is no client-side router: each screen is a
real URL that works with the back button and can be linked to.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, tzinfo
from typing import Annotated, Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response

from dentist_ai.api.cookies import csrf_session_id
from dentist_ai.api.deps import AppSettings, OptionalUser, Sessions, get_settings
from dentist_ai.clinical.labels import volume_disclaimer
from dentist_ai.core.security import SessionService
from dentist_ai.db.models import User
from dentist_ai.web.templating import build_templates

PageHandler = Callable[..., Coroutine[None, None, HTMLResponse]]

router = APIRouter(include_in_schema=False)

_MARKETING_PAGES: dict[str, tuple[str, str, str]] = {
    # route name -> (template, <title>, meta description)
    "landing": (
        "marketing/landing.html",
        "Dentist-AI — ИИ-анализ дентальных снимков",
        "Нейросеть находит патологии на рентгеновских снимках за секунды: "
        "31 класс находок, вероятность по каждой, полная карта пациента.",
    ),
    "about": (
        "marketing/about.html",
        "О продукте — Dentist-AI",
        "Как устроен Dentist-AI: модель детекции, метрики качества и принципы "
        "работы с медицинскими данными.",
    ),
    "pricing": (
        "marketing/pricing.html",
        "Тарифы — Dentist-AI",
        "Прозрачные тарифы для клиник любого размера. Бесплатный старт, без карты.",
    ),
    "contact": (
        "marketing/contact.html",
        "Контакты — Dentist-AI",
        "Свяжитесь с командой Dentist-AI: демо, внедрение, поддержка.",
    ),
    "privacy": (
        "marketing/privacy.html",
        "Политика конфиденциальности — Dentist-AI",
        "Как Dentist-AI собирает, хранит и защищает данные клиник и пациентов.",
    ),
}


def _csrf_token(request: Request, sessions: SessionService) -> str:
    """Mint a CSRF token bound to whatever ``CSRFMiddleware`` will check against.

    Anonymous visitors (login/register forms) get one bound to a host-derived
    pseudo-session, which is re-derived identically when the POST arrives.
    """
    return sessions.issue_csrf(csrf_session_id(request, sessions, get_settings(request)))


def _render(
    request: Request,
    template: str,
    *,
    title: str,
    description: str = "",
    user: User | None = None,
    csrf_token: str = "",
    **context: object,
) -> HTMLResponse:
    settings = get_settings(request)
    templates = build_templates(settings)
    return templates.TemplateResponse(
        request=request,
        name=template,
        context={
            "title": title,
            "description": description,
            "user": user,
            "csrf_token": csrf_token,
            "current_path": request.url.path,
            **context,
        },
    )


# --------------------------------------------------------------------------
# Marketing
# --------------------------------------------------------------------------
def _marketing_route(key: str) -> PageHandler:
    template, title, description = _MARKETING_PAGES[key]

    async def handler(request: Request, user: OptionalUser) -> HTMLResponse:
        return _render(request, template, title=title, description=description, user=user)

    return handler


router.add_api_route("/", _marketing_route("landing"), methods=["GET"])
router.add_api_route("/about", _marketing_route("about"), methods=["GET"])
router.add_api_route("/pricing", _marketing_route("pricing"), methods=["GET"])
router.add_api_route("/contact", _marketing_route("contact"), methods=["GET"])
router.add_api_route("/privacy", _marketing_route("privacy"), methods=["GET"])


# --------------------------------------------------------------------------
# Auth screens
# --------------------------------------------------------------------------
@router.get("/login")
async def login_page(request: Request, user: OptionalUser, sessions: Sessions) -> Response:
    if user is not None:
        return RedirectResponse("/app", status_code=303)
    return _render(
        request,
        "auth/login.html",
        title="Вход — Dentist-AI",
        description="Войдите в рабочее пространство вашей клиники.",
        csrf_token=_csrf_token(request, sessions),
    )


@router.get("/register")
async def register_page(request: Request, user: OptionalUser, sessions: Sessions) -> Response:
    if user is not None:
        return RedirectResponse("/app", status_code=303)
    return _render(
        request,
        "auth/register.html",
        title="Регистрация — Dentist-AI",
        description="Создайте аккаунт клиники за минуту. Карта не нужна.",
        csrf_token=_csrf_token(request, sessions),
    )


# --------------------------------------------------------------------------
# Application shell
# --------------------------------------------------------------------------
class LoginRequiredError(Exception):
    """Raised by page dependencies for anonymous visitors.

    A browser hitting a protected *page* should land on the sign-in screen,
    not receive a JSON 401 — but the same dependency chain serves the API,
    where a redirect would be wrong. Signalling with an exception keeps that
    decision in one handler (registered in ``create_app``) instead of
    scattering ``if user is None`` across every route.
    """

    def __init__(self, next_path: str) -> None:
        self.next_path = next_path
        super().__init__(next_path)


async def _require_page_user(user: OptionalUser, request: Request) -> User:
    if user is None:
        raise LoginRequiredError(request.url.path)
    return user


PageUser = Annotated[User, Depends(_require_page_user)]

_APP_PAGES: dict[str, tuple[str, str]] = {
    "/app/studies": ("app/studies.html", "Снимки"),
    "/app/volumes": ("app/volumes.html", "КЛКТ"),
    "/app/patients": ("app/patients.html", "Пациенты"),
    "/app/settings": ("app/settings.html", "Настройки"),
}

#: Russian month names in the genitive case — "29 июля", not "29 июль".
#: `strftime('%B')` would return whichever locale the container happens to
#: have installed, which in a slim image is C.
_MONTHS_GENITIVE: Final[tuple[str, ...]] = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)
_WEEKDAYS: Final[tuple[str, ...]] = (
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
)

#: Lower bound of each greeting, in local hours. Rendered on the server so the
#: page does not flash a neutral greeting before JavaScript corrects it.
_GREETINGS: Final[tuple[tuple[int, str], ...]] = (
    (23, "Доброй ночи"),
    (18, "Добрый вечер"),
    (12, "Добрый день"),
    (5, "Доброе утро"),
    (0, "Доброй ночи"),
)


def _clinic_now(timezone: str) -> datetime:
    """The wall clock the clinic's staff are reading."""
    try:
        zone: tzinfo = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        zone = UTC
    return datetime.now(zone)


def _greeting(now: datetime) -> str:
    return next(label for hour, label in _GREETINGS if now.hour >= hour)


def _long_date(now: datetime) -> str:
    return f"{_WEEKDAYS[now.weekday()]}, {now.day} {_MONTHS_GENITIVE[now.month - 1]}"


@router.get("/app")
async def dashboard_page(request: Request, user: PageUser, sessions: Sessions) -> HTMLResponse:
    """The dashboard, which unlike the other app pages is addressed to a person.

    The greeting and the date are resolved against the organisation's timezone
    rather than the server's: a clinic in Almaty should not be wished good
    evening because the container runs in UTC.
    """
    now = _clinic_now(user.organization.timezone)
    return _render(
        request,
        "app/dashboard.html",
        title="Обзор — Dentist-AI",
        user=user,
        csrf_token=_csrf_token(request, sessions),
        page_title="Обзор",
        greeting=_greeting(now),
        today_label=_long_date(now),
    )


def _app_route(path: str) -> PageHandler:
    template, title = _APP_PAGES[path]

    async def handler(request: Request, user: PageUser, sessions: Sessions) -> HTMLResponse:
        return _render(
            request,
            template,
            title=f"{title} — Dentist-AI",
            user=user,
            csrf_token=_csrf_token(request, sessions),
            page_title=title,
        )

    return handler


for _path in _APP_PAGES:
    router.add_api_route(_path, _app_route(_path), methods=["GET"])


@router.get("/app/studies/{public_id}")
async def study_detail_page(
    request: Request, public_id: str, user: PageUser, sessions: Sessions
) -> HTMLResponse:
    return _render(
        request,
        "app/study_detail.html",
        title="Снимок — Dentist-AI",
        user=user,
        csrf_token=_csrf_token(request, sessions),
        page_title="Снимок",
        public_id=public_id,
    )


@router.get("/app/patients/{patient_id}")
async def patient_detail_page(
    request: Request, patient_id: int, user: PageUser, sessions: Sessions
) -> HTMLResponse:
    return _render(
        request,
        "app/patient_detail.html",
        title="Пациент — Dentist-AI",
        user=user,
        csrf_token=_csrf_token(request, sessions),
        page_title="Пациент",
        patient_id=patient_id,
    )


@router.get("/app/volumes/{public_id}")
async def volume_detail_page(
    request: Request, public_id: str, user: PageUser, sessions: Sessions
) -> HTMLResponse:
    """The CBCT workstation.

    The disclaimer is rendered server-side rather than by the viewer bundle: it
    has to be in the document even if the JavaScript fails to load, because a
    page that shows a patient's scan without it is a page making a claim it is
    not entitled to make.
    """
    return _render(
        request,
        "app/volume_detail.html",
        title="КЛКТ — Dentist-AI",
        user=user,
        csrf_token=_csrf_token(request, sessions),
        page_title="КЛКТ",
        public_id=public_id,
        volume_disclaimer=volume_disclaimer(user.locale),
    )


@router.get("/app/scans/{public_id}")
async def scan_detail_page(
    request: Request, public_id: str, user: PageUser, sessions: Sessions
) -> HTMLResponse:
    return _render(
        request,
        "app/scan_detail.html",
        title="3D-модель — Dentist-AI",
        user=user,
        csrf_token=_csrf_token(request, sessions),
        page_title="3D-модель",
        public_id=public_id,
    )


# --------------------------------------------------------------------------
# Crawler files
# --------------------------------------------------------------------------
@router.get("/robots.txt", response_class=PlainTextResponse)
async def robots(request: Request, settings: AppSettings) -> PlainTextResponse:
    base = str(request.base_url).rstrip("/")
    if not settings.is_production:
        # Never let a staging deployment get indexed.
        return PlainTextResponse("User-agent: *\nDisallow: /\n")
    return PlainTextResponse(
        "User-agent: *\n"
        "Allow: /$\n"
        "Allow: /about\n"
        "Allow: /pricing\n"
        "Allow: /contact\n"
        "Allow: /privacy\n"
        # The application itself holds patient data; keep it out of every index.
        "Disallow: /app\n"
        "Disallow: /api\n"
        f"\nSitemap: {base}/sitemap.xml\n"
    )


@router.get("/sitemap.xml")
async def sitemap(request: Request) -> Response:
    base = str(request.base_url).rstrip("/")
    urls = "".join(
        f"<url><loc>{base}{path}</loc><changefreq>weekly</changefreq>"
        f"<priority>{priority}</priority></url>"
        for path, priority in (
            ("/", "1.0"),
            ("/about", "0.8"),
            ("/pricing", "0.9"),
            ("/contact", "0.6"),
            ("/privacy", "0.3"),
        )
    )
    return Response(
        content=(
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            f"{urls}</urlset>"
        ),
        media_type="application/xml",
    )
