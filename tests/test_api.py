"""API tests: valid input, missing fields, wrong types, out-of-range values, unseen categories."""
from __future__ import annotations

import pytest

FIELDS = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]


# ---------- service endpoints ----------
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert 0 < body["threshold"] < 1 + 1e-9
    from src.features import INPUT_FEATURES

    assert body["input_features"] == INPUT_FEATURES


def test_docs_available(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


# ---------- valid input ----------
def test_predict_valid(client, valid_txn):
    r = client.post("/predict", json=valid_txn)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {"is_fraud", "prediction", "fraud_probability", "threshold", "model_name"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["is_fraud"] == bool(body["prediction"])
    assert body["prediction"] == int(body["fraud_probability"] >= body["threshold"])
    assert "X-Process-Time-ms" in r.headers


def test_fraud_pattern_scores_higher_than_legit(client, valid_txn, legit_txn):
    p_fraud = client.post("/predict", json=valid_txn).json()["fraud_probability"]
    p_legit = client.post("/predict", json=legit_txn).json()["fraud_probability"]
    assert p_fraud > p_legit


@pytest.mark.parametrize("ttype", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
def test_all_known_types_accepted(client, valid_txn, ttype):
    assert client.post("/predict", json={**valid_txn, "type": ttype}).status_code == 200


def test_integer_amount_coerced(client, valid_txn):
    assert client.post("/predict", json={**valid_txn, "amount": 181000}).status_code == 200


# ---------- missing fields ----------
@pytest.mark.parametrize("field", FIELDS)
def test_missing_field_rejected(client, valid_txn, field):
    payload = {k: v for k, v in valid_txn.items() if k != field}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert any(field in str(e["loc"]) for e in r.json()["detail"])


def test_empty_body_rejected(client):
    assert client.post("/predict", json={}).status_code == 422


# ---------- wrong data types ----------
@pytest.mark.parametrize("field,value", [
    ("amount", "lots"), ("step", "ten"), ("step", 3.7), ("oldbalanceOrg", [1, 2]),
    ("newbalanceDest", {"v": 1}), ("type", 5),
])
def test_wrong_type_rejected(client, valid_txn, field, value):
    assert client.post("/predict", json={**valid_txn, field: value}).status_code == 422


def test_non_json_body_rejected(client):
    r = client.post("/predict", content="not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


# ---------- out-of-range values ----------
@pytest.mark.parametrize("field,value", [
    ("amount", 0), ("amount", -10.0), ("amount", 1e12), ("step", 0), ("step", -5),
    ("oldbalanceOrg", -1.0), ("newbalanceOrig", -0.01), ("oldbalanceDest", -100.0),
    ("newbalanceDest", 1e11),
])
def test_out_of_range_rejected(client, valid_txn, field, value):
    assert client.post("/predict", json={**valid_txn, field: value}).status_code == 422


# ---------- unseen categories / unexpected fields ----------
@pytest.mark.parametrize("ttype", ["BITCOIN", "transfer", "", "WIRE"])
def test_unseen_category_rejected(client, valid_txn, ttype):
    r = client.post("/predict", json={**valid_txn, "type": ttype})
    assert r.status_code == 422


def test_unexpected_field_rejected(client, valid_txn):
    assert client.post("/predict", json={**valid_txn, "isFlaggedFraud": 1}).status_code == 422


# ---------- batch ----------
def test_batch(client, valid_txn, legit_txn):
    r = client.post("/predict/batch", json={"transactions": [valid_txn, legit_txn, valid_txn]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3 and len(body["predictions"]) == 3
    assert body["fraud_count"] == sum(p["prediction"] for p in body["predictions"])


def test_batch_empty_rejected(client):
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422


def test_batch_too_large_rejected(client, valid_txn):
    assert client.post("/predict/batch", json={"transactions": [valid_txn] * 1001}).status_code == 422


def test_batch_with_one_bad_item_rejected(client, valid_txn):
    bad = {**valid_txn, "type": "BITCOIN"}
    assert client.post("/predict/batch", json={"transactions": [valid_txn, bad]}).status_code == 422


# ---------- Azure client endpoint ----------
def test_azure_not_configured_returns_503(client, valid_txn):
    assert client.post("/predict/azure", json=valid_txn).status_code == 503


def test_azure_proxied(client, valid_txn, monkeypatch):
    from api import azure_client

    captured = {}

    def fake_score(records, timeout=30.0):
        captured["records"] = records
        return [{"fraud_probability": 0.91, "is_fraud": 1}]

    monkeypatch.setattr(azure_client, "score", fake_score)
    r = client.post("/predict/azure", json=valid_txn)
    assert r.status_code == 200
    assert r.json()["is_fraud"] is True and r.json()["fraud_probability"] == 0.91
    assert captured["records"][0]["type"] == "TRANSFER"


def test_azure_payload_and_parsing():
    from api.azure_client import build_payload, parse_response

    rec = {"step": 1, "type": "TRANSFER", "amount": 10.0, "oldbalanceOrg": 10.0, "newbalanceOrig": 0.0,
           "oldbalanceDest": 0.0, "newbalanceDest": 0.0}
    p = build_payload([rec])["input_data"]
    assert p["columns"][0] == "step" and len(p["data"][0]) == 7
    assert parse_response([{"fraud_probability": 0.2, "is_fraud": 0}])[0]["is_fraud"] == 0
    assert parse_response({"fraud_probability": [0.7], "is_fraud": [1]})[0]["fraud_probability"] == 0.7
