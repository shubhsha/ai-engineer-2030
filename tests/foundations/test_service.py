import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from ai_engineer.foundations.idempotency import IdempotencyInProgress
from ai_engineer.service.app import create_app

BODY = {"items": [{"price": "19.99", "qty": 2}, {"price": "5", "qty": 1}]}


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_retry_with_same_key_replays_the_first_response(client):
    headers = {"Idempotency-Key": "order-7"}
    first = client.post("/v1/quotes", json=BODY, headers=headers)
    second = client.post("/v1/quotes", json=BODY, headers=headers)
    assert first.status_code == 201
    assert second.status_code == 200
    assert second.headers["Idempotent-Replayed"] == "true"
    assert first.json() == second.json()
    assert first.json()["total"] == "44.98"


def test_missing_key_is_a_problem_document(client):
    response = client.post("/v1/quotes", json=BODY)
    assert response.status_code == 400
    assert response.headers["content-type"].startswith(
        "application/problem+json"
    )
    assert response.json()["title"] == "Missing Idempotency-Key"


def test_same_key_different_body_is_rejected(client):
    headers = {"Idempotency-Key": "order-8"}
    client.post("/v1/quotes", json=BODY, headers=headers)
    other = client.post("/v1/quotes", json={"items": []}, headers=headers)
    assert other.status_code == 422


def test_invalid_items_are_reported(client):
    bad = {"items": [{"price": "free", "qty": 1}]}
    response = client.post(
        "/v1/quotes", json=bad, headers={"Idempotency-Key": "k"}
    )
    assert response.status_code == 422  # understood, but cannot be quoted
    assert response.json()["type"].endswith("/invalid-quote")


class BusyStore:
    """A fake store whose first attempt for every key is still running."""

    def execute(self, key, payload, handler):
        raise IdempotencyInProgress(key)


def test_a_key_still_in_use_is_409_with_retry_after():
    client = TestClient(create_app(store=BusyStore()))
    response = client.post(
        "/v1/quotes", json=BODY, headers={"Idempotency-Key": "order-9"}
    )
    assert response.status_code == 409
    assert response.headers["Retry-After"] == "1"
    assert response.json()["title"] == "Request in progress"


KEYED = {"Content-Type": "application/json", "Idempotency-Key": "k"}
FAILURES = [
    ("GET", "/v1/nothing", None, 404),
    ("GET", "/v1/quotes", None, 405),
    ("POST", "/v1/quotes", b'{"items": [}', 400),
    ("POST", "/v1/quotes", b"[1, 2]", 422),
]


@pytest.mark.parametrize(("method", "path", "content", "status"), FAILURES)
def test_every_failure_is_a_problem_document(
    client, method, path, content, status
):
    response = client.request(method, path, content=content, headers=KEYED)
    assert response.status_code == status
    assert response.headers["content-type"].startswith(
        "application/problem+json"
    )
    document = response.json()
    assert document["status"] == status
    assert {"type", "title", "detail"} <= document.keys()


def test_a_405_still_says_which_methods_are_allowed(client):
    assert client.get("/v1/quotes").headers["Allow"] == "POST"


def test_validation_errors_name_the_field_but_never_echo_the_input(client):
    response = client.post("/v1/quotes", json=["card-4111"], headers=KEYED)
    assert response.status_code == 422
    assert "card-4111" not in response.text
    assert response.json()["errors"][0]["loc"] == ["body"]


class ExplodingStore:
    def execute(self, key, payload, handler):
        raise RuntimeError("db password=hunter2")


def test_a_server_bug_is_a_500_that_keeps_its_cause_in_the_log(caplog):
    client = TestClient(
        create_app(store=ExplodingStore()), raise_server_exceptions=False
    )
    response = client.post("/v1/quotes", json=BODY, headers=KEYED)
    assert response.status_code == 500
    assert "hunter2" not in response.text
    incident = response.json()["instance"]
    assert incident.startswith("urn:uuid:")
    assert incident in caplog.text  # support can find the traceback
    assert "hunter2" in caplog.text


def test_the_document_lists_every_status_the_service_returns(client):
    document = client.get("/openapi.json").json()
    documented = set(document["paths"]["/v1/quotes"]["post"]["responses"])
    busy = TestClient(create_app(store=BusyStore()))
    seen = {
        client.post("/v1/quotes", json=BODY).status_code,  # no key
        client.post("/v1/quotes", content=b"{", headers=KEYED).status_code,
        client.post("/v1/quotes", json=BODY, headers=KEYED).status_code,
        client.post("/v1/quotes", json=BODY, headers=KEYED).status_code,
        client.post("/v1/quotes", json={}, headers=KEYED).status_code,
        busy.post("/v1/quotes", json=BODY, headers=KEYED).status_code,
    }
    assert seen == {200, 201, 400, 409, 422}
    assert {str(status) for status in seen} <= documented


# What the checkout client reads from a quote, and nothing more.
CHECKOUT = {"status": 201, "reads": {"quote_id": str, "total": str}}


def breaks(reply, status):
    """Which of the checkout client's expectations this reply breaks."""
    broken = [] if status == CHECKOUT["status"] else [f"status {status}"]
    for name, kind in CHECKOUT["reads"].items():
        if not isinstance(reply.get(name), kind):
            broken.append(name)
    return broken


def test_the_service_keeps_the_checkout_contract(client):
    response = client.post("/v1/quotes", json=BODY, headers=KEYED)
    assert breaks(response.json(), response.status_code) == []
    assert response.json()["total"] == "44.98"  # meaning, not just shape


def test_an_added_field_does_not_break_the_checkout_client():
    reply = {"quote_id": "cbc34b81e817", "total": "10", "currency": "USD"}
    assert breaks(reply, 201) == []


@pytest.mark.parametrize(
    ("reply", "broken"),
    [
        ({"total": "10"}, ["quote_id"]),  # a field was removed
        ({"quote_id": "c", "amount": "10"}, ["total"]),  # renamed
        ({"quote_id": "c", "total": 10}, ["total"]),  # retyped
    ],
)
def test_a_removed_renamed_or_retyped_field_breaks_it(reply, broken):
    assert breaks(reply, 201) == broken
