from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "call_failure": 8,
    "complains": 1,
    "subscription_length": 20,
    "charge_amount": 3,
    "seconds_of_use": 2500,
    "frequency_of_use": 55,
    "frequency_of_sms": 12,
    "distinct_called_numbers": 25,
    "age_group": 3,
    "tariff_plan": 1,
    "status": 1,
    "age": 42,
    "customer_value": 1200.5,
}


def test_health_has_version_metadata() -> None:
    response = client.get("/health")
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "ok"
    assert "model_version" in body
    assert "api_version" in body


def test_version_endpoint() -> None:
    response = client.get("/version")
    body = response.json()

    assert response.status_code == 200
    assert body["api_version"]
    assert body["model_version"]


def test_valid_prediction_contract() -> None:
    response = client.post("/predictions", json=VALID_PAYLOAD)
    body = response.json()

    assert response.status_code == 200
    assert body["prediction"] in (0, 1)
    assert body["model_version"]
    assert body["request_id"]


def test_probability_is_between_zero_and_one() -> None:
    body = client.post("/predictions", json=VALID_PAYLOAD).json()

    assert 0 <= body["churn_probability"] <= 1


def test_negative_input_is_rejected() -> None:
    payload = {**VALID_PAYLOAD, "call_failure": -1}
    response = client.post("/predictions", json=payload)

    assert response.status_code == 422


def test_missing_required_field_is_rejected() -> None:
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
    response = client.post("/predictions", json=payload)

    assert response.status_code == 422
