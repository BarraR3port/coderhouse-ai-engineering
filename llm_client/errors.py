import ssl

import anthropic
import anyio
import httpx2
import openai

from llm_client.schemas import ErrorInfo, ErrorKind, Provider


class ConfigurationError(ValueError):
    pass


class LLMError(Exception):
    def __init__(self, info: ErrorInfo) -> None:
        super().__init__(f"[{info.provider}] {info.kind}: {info.message}")
        self.info = info


type ProviderFailure = (
    openai.APIError | anthropic.APIError | httpx2.RequestError | ssl.SSLError | anyio.EndOfStream
)

PROVIDER_FAILURES = (
    openai.APIError,
    anthropic.APIError,
    httpx2.RequestError,
    ssl.SSLError,
    anyio.EndOfStream,
)

ERROR_TYPE_KINDS = {
    "insufficient_quota": ErrorKind.QUOTA_EXCEEDED,
    "billing_error": ErrorKind.QUOTA_EXCEEDED,
    "rate_limit_error": ErrorKind.RATE_LIMIT,
    "rate_limit_exceeded": ErrorKind.RATE_LIMIT,
    "overloaded_error": ErrorKind.PROVIDER_UNAVAILABLE,
    "api_error": ErrorKind.PROVIDER_UNAVAILABLE,
    "server_error": ErrorKind.PROVIDER_UNAVAILABLE,
    "timeout_error": ErrorKind.TIMEOUT,
    "authentication_error": ErrorKind.AUTHENTICATION,
    "permission_error": ErrorKind.AUTHENTICATION,
    "invalid_request_error": ErrorKind.INVALID_REQUEST,
    "not_found_error": ErrorKind.INVALID_REQUEST,
    "request_too_large": ErrorKind.INVALID_REQUEST,
}


def classify_sdk_error(error: ProviderFailure, provider: Provider) -> ErrorInfo:
    match error:
        case httpx2.TimeoutException() | openai.APITimeoutError() | anthropic.APITimeoutError():
            return ErrorInfo(kind=ErrorKind.TIMEOUT, provider=provider, message=describe(error))
        case (
            httpx2.RequestError()
            | ssl.SSLError()
            | anyio.EndOfStream()
            | openai.APIConnectionError()
            | anthropic.APIConnectionError()
        ):
            return ErrorInfo(kind=ErrorKind.CONNECTION, provider=provider, message=describe(error))
        case openai.APIStatusError() | anthropic.APIStatusError() if error.status_code < 400:
            return ErrorInfo(
                kind=kind_for_error_body(error.body), provider=provider, message=error.message
            )
        case openai.APIStatusError() | anthropic.APIStatusError():
            return ErrorInfo(
                kind=kind_for_status_error(error.status_code, error.body),
                provider=provider,
                message=error.message,
                status_code=error.status_code,
                retry_after_seconds=retry_after_seconds(error.response.headers),
            )
        case openai.APIError() | anthropic.APIError():
            return ErrorInfo(
                kind=kind_for_error_body(error.body), provider=provider, message=error.message
            )
        case _:
            return ErrorInfo(kind=ErrorKind.UNKNOWN, provider=provider, message=describe(error))


def kind_for_status_error(status_code: int, body: object) -> ErrorKind:
    if kind_for_error_body(body) is ErrorKind.QUOTA_EXCEEDED:
        return ErrorKind.QUOTA_EXCEEDED
    return kind_for_status(status_code)


def kind_for_error_body(body: object) -> ErrorKind:
    if not isinstance(body, dict):
        return ErrorKind.UNKNOWN
    details = body.get("error")
    source = details if isinstance(details, dict) else body
    for key in ("code", "type"):
        if (kind := ERROR_TYPE_KINDS.get(str(source.get(key)))) is not None:
            return kind
    return ErrorKind.UNKNOWN


def kind_for_status(status_code: int) -> ErrorKind:
    match status_code:
        case 401 | 403:
            return ErrorKind.AUTHENTICATION
        case 402:
            return ErrorKind.QUOTA_EXCEEDED
        case 408:
            return ErrorKind.TIMEOUT
        case 429:
            return ErrorKind.RATE_LIMIT
        case 409:
            return ErrorKind.PROVIDER_UNAVAILABLE
        case code if code >= 500:
            return ErrorKind.PROVIDER_UNAVAILABLE
        case code if 400 <= code < 500:
            return ErrorKind.INVALID_REQUEST
        case _:
            return ErrorKind.UNKNOWN


def retry_after_seconds(headers: httpx2.Headers) -> float | None:
    for header, scale in (("retry-after-ms", 1000), ("retry-after", 1)):
        try:
            return max(float(headers[header]) / scale, 0.0)
        except (KeyError, ValueError):
            continue
    return None


def describe(error: Exception) -> str:
    if isinstance(error, openai.APIError | anthropic.APIError):
        return error.message
    return str(error) or f"la conexión se cortó ({type(error).__name__})"
