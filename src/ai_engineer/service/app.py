"""Chapter 9: ``POST /v1/quotes`` with idempotency keys and problem details.

Run it with ``uvicorn ai_engineer.service.app:app`` (``api`` extra).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from http import HTTPStatus
from typing import Annotated, Any

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ai_engineer.foundations.idempotency import (
    JSON,
    IdempotencyBackend,
    IdempotencyConflict,
    IdempotencyInProgress,
    IdempotencyStore,
    fingerprint,
    problem,
)

log = logging.getLogger(__name__)

PROBLEM = "application/problem+json"
TYPES = "https://api.example.com/problems/"

_QUOTE: JSON = {
    "type": "object",
    "required": ["quote_id", "total"],
    "properties": {
        "quote_id": {"type": "string"},
        "total": {"type": "string", "description": "A decimal, e.g. 44.98."},
    },
}
_PROBLEM: JSON = {
    "type": "object",
    "required": ["type", "title", "status"],
    "properties": {
        "type": {"type": "string", "format": "uri-reference"},
        "title": {"type": "string"},
        "status": {"type": "integer"},
        "detail": {"type": "string"},
        "instance": {"type": "string", "format": "uri-reference"},
    },
}
_ITEM: JSON = {
    "type": "object",
    "required": ["price", "qty"],
    "properties": {
        "price": {"type": ["string", "number"]},
        "qty": {"type": "integer"},
    },
}
_REQUEST: JSON = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": _ITEM}},
}


def _says(description: str, schema: JSON, media: str = PROBLEM) -> JSON:
    return {"description": description, "content": {media: {"schema": schema}}}


RESPONSES: dict[int | str, JSON] = {
    200: _says("The first response, replayed", _QUOTE, "application/json"),
    201: _says("The quote, created", _QUOTE, "application/json"),
    400: _says("No Idempotency-Key, or the body is not JSON", _PROBLEM),
    409: _says("The first attempt with this key is still running", _PROBLEM),
    422: _says("Key reused with another body, or an invalid quote", _PROBLEM),
}


def _quote(body: JSON) -> JSON:
    try:
        total = sum(
            (
                Decimal(str(item["price"])) * int(item["qty"])
                for item in body.get("items", [])
            ),
            Decimal(0),
        )
    except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
        raise ValueError("each item needs a numeric price and qty") from exc
    return {"quote_id": fingerprint(body)[:12], "total": str(total)}


def _problem(
    status: int,
    title: str,
    detail: str,
    slug: str | None = None,
    headers: Mapping[str, str] | None = None,
    **extra: Any,
) -> JSONResponse:
    """A problem document; ``slug`` names a type, none means about:blank."""
    type_uri = "about:blank" if slug is None else TYPES + slug
    return JSONResponse(
        problem(status, title, detail, type_uri, **extra),
        status_code=status,
        media_type=PROBLEM,
        headers=headers,
    )


def create_app(store: IdempotencyBackend | None = None) -> FastAPI:
    app = FastAPI(title="ai-engineer-2030 demo service")
    keys = store if store is not None else IdempotencyStore()

    # region quote-endpoint
    @app.post(
        "/v1/quotes",
        status_code=201,
        responses=RESPONSES,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": _REQUEST}},
            }
        },
    )
    def create_quote(
        body: dict[str, Any],
        idempotency_key: Annotated[str | None, Header()] = None,
    ) -> JSONResponse:
        if not idempotency_key:
            return _problem(
                400,
                "Missing Idempotency-Key",
                "Send an Idempotency-Key header so retries are safe.",
                "missing-idempotency-key",
            )
        try:
            response, replayed = keys.execute(
                idempotency_key, body, lambda: _quote(body)
            )
        except IdempotencyConflict:
            return _problem(
                422,
                "Idempotency-Key reused",
                "This key was already used with a different body.",
                "idempotency-key-reused",
            )
        except IdempotencyInProgress:
            return _problem(
                409,
                "Request in progress",
                "The first request with this key is still running. "
                "Retry after a second to get its response.",
                "request-in-progress",
                headers={"Retry-After": "1"},
            )
        except ValueError as exc:
            return _problem(422, "Invalid quote", str(exc), "invalid-quote")
        return JSONResponse(
            response,
            status_code=200 if replayed else 201,
            headers={"Idempotent-Replayed": str(replayed).lower()},
        )

    # endregion

    @app.exception_handler(StarletteHTTPException)
    def no_such_route(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _problem(
            exc.status_code,
            HTTPStatus(exc.status_code).phrase,
            "The paths and methods on offer are listed in /openapi.json.",
            headers=exc.headers,
        )

    # region problem-handlers
    @app.exception_handler(RequestValidationError)
    def bad_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        found = exc.errors()
        syntax = any(e["type"] == "json_invalid" for e in found)
        return _problem(
            400 if syntax else 422,
            "Malformed JSON" if syntax else "Invalid request",
            "Fix the fields listed under errors and send it again.",
            "malformed-json" if syntax else "invalid-request",
            errors=[{"loc": list(e["loc"]), "msg": e["msg"]} for e in found],
        )

    @app.exception_handler(Exception)
    def server_bug(_: Request, exc: Exception) -> JSONResponse:
        incident = f"urn:uuid:{uuid.uuid4()}"
        log.error("unhandled error, instance %s", incident, exc_info=exc)
        return _problem(
            500,
            HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
            "The server failed. Quote the instance when you report it.",
            instance=incident,
        )

    # endregion

    return app


app = create_app()
