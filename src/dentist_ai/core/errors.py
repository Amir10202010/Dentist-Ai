"""Application error taxonomy and RFC 9457 problem responses.

Every message a client sees is one authored here. Anything else becomes an
opaque 500, with the details going only to the logs.
"""

from __future__ import annotations

from typing import ClassVar

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from dentist_ai.core.logging import get_logger

log = get_logger(__name__)

PROBLEM_CONTENT_TYPE = "application/problem+json"


class AppError(Exception):
    """Base class for every error that is safe to show a client.

    The ``code`` values are mirrored by the ``ApiErrorCode`` union in
    ``frontend/src/lib/api.ts``.
    """

    status_code: ClassVar[int] = status.HTTP_400_BAD_REQUEST
    code: ClassVar[str] = "bad_request"
    default_message: ClassVar[str] = "Запрос не может быть обработан."

    def __init__(
        self,
        message: str | None = None,
        *,
        field_errors: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.field_errors = field_errors or {}
        self.headers = headers or {}
        super().__init__(self.message)

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": f"https://dentist-ai.app/errors/{self.code}",
            "title": self.message,
            "status": self.status_code,
            "code": self.code,
        }
        if self.field_errors:
            payload["errors"] = self.field_errors
        return payload


class ValidationError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "validation_failed"
    default_message = "Проверьте правильность заполнения полей."


class AuthenticationError(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "unauthenticated"
    default_message = "Необходимо войти в аккаунт."


class InvalidCredentialsError(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "invalid_credentials"
    # Identical for unknown-email and wrong-password, so the endpoint cannot
    # be used to enumerate registered addresses.
    default_message = "Неверный email или пароль."


class PermissionDeniedError(AppError):
    status_code = status.HTTP_403_FORBIDDEN
    code = "permission_denied"
    default_message = "Недостаточно прав для этого действия."


class CSRFError(AppError):
    status_code = status.HTTP_403_FORBIDDEN
    code = "csrf_failed"
    default_message = "Сессия устарела. Обновите страницу и попробуйте снова."


class NotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"
    default_message = "Ресурс не найден."


class MethodNotAllowedError(AppError):
    status_code = status.HTTP_405_METHOD_NOT_ALLOWED
    code = "method_not_allowed"
    default_message = "Метод не поддерживается."


class ConflictError(AppError):
    status_code = status.HTTP_409_CONFLICT
    code = "conflict"
    default_message = "Ресурс уже существует."


class EmailAlreadyRegisteredError(ConflictError):
    code = "email_taken"
    default_message = "Этот email уже зарегистрирован."


class PayloadTooLargeError(AppError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    code = "payload_too_large"
    default_message = "Файл слишком большой."


class UnsupportedMediaTypeError(AppError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    code = "unsupported_media_type"
    default_message = "Неподдерживаемый формат файла."


class RateLimitedError(AppError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "rate_limited"
    default_message = "Слишком много попыток. Попробуйте позже."

    def __init__(self, retry_after_seconds: int, message: str | None = None) -> None:
        super().__init__(message, headers={"Retry-After": str(retry_after_seconds)})
        self.retry_after_seconds = retry_after_seconds


class InferenceError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "inference_unavailable"
    default_message = "Модель анализа временно недоступна. Попробуйте позже."


class InternalError(AppError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "internal_error"
    default_message = "Внутренняя ошибка сервера."


def problem_response(error: AppError) -> JSONResponse:
    """Render an ``AppError`` as an RFC 9457 problem document.

    Public because middleware needs it: exceptions raised inside Starlette
    middleware never reach FastAPI's exception handlers — those only wrap the
    router — so middleware must build the response itself rather than raise.
    """
    return JSONResponse(
        status_code=error.status_code,
        content=error.to_payload(),
        headers=error.headers,
        media_type=PROBLEM_CONTENT_TYPE,
    )


_problem_response = problem_response


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers that guarantee a uniform error shape everywhere."""

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
            log.error("app_error", code=exc.code, message=exc.message)
        else:
            log.info("client_error", code=exc.code, status=exc.status_code)
        return _problem_response(exc)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        field_errors: dict[str, str] = {}
        for item in exc.errors():
            location = [str(part) for part in item["loc"] if part not in ("body", "query")]
            field_errors[".".join(location) or "_"] = str(item["msg"])
        return _problem_response(ValidationError(field_errors=field_errors))

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        factory: type[AppError] = {
            status.HTTP_401_UNAUTHORIZED: AuthenticationError,
            status.HTTP_403_FORBIDDEN: PermissionDeniedError,
            status.HTTP_404_NOT_FOUND: NotFoundError,
            status.HTTP_405_METHOD_NOT_ALLOWED: MethodNotAllowedError,
        }.get(
            exc.status_code,
            InternalError if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR else AppError,
        )
        return _problem_response(factory())

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        # The only place a raw exception is allowed to surface — to the log,
        # never to the response body.
        log.exception("unhandled_exception", error_type=type(exc).__name__)
        return _problem_response(InternalError())
