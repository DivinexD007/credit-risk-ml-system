"""
tests/test_api.py
-----------------
Integration tests for the FastAPI inference service.

Uses FastAPI's TestClient (built on httpx) to test all endpoints without
starting a real server. The tests mock the model predictor to avoid
requiring a trained model file on disk during CI/CD.

Run with
--------
    pytest tests/test_api.py -v
    pytest tests/ -v --tb=short

Fixtures
--------
- client         : TestClient with real predictor (requires trained model)
- client_mocked  : TestClient with mocked predictor (no model file needed)
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------
VALID_APPLICATION = {
    "Age": 35.0,
    "Income": 55000.0,
    "LoanAmount": 15000.0,
    "CreditScore": 720.0,
    "MonthsEmployed": 36.0,
    "NumCreditLines": 3,
    "InterestRate": 7.5,
    "LoanTerm": 36,
    "EmploymentType": "Full-time",
}

HIGH_RISK_APPLICATION = {
    "Age": 22.0,
    "Income": 18000.0,
    "LoanAmount": 25000.0,
    "CreditScore": 310.0,
    "MonthsEmployed": 1.0,
    "NumCreditLines": 0,
    "InterestRate": 24.9,
    "LoanTerm": 60,
    "EmploymentType": "Unemployed",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_predictor():
    """
    Create a mock CreditRiskPredictor that returns a deterministic result.
    Used when the trained model file is not available (CI/CD environments).
    """
    predictor = MagicMock()
    predictor.predict.return_value = {
        "prediction":          1,
        "probability_approve": 0.82,
        "probability_default": 0.18,
        "risk_tier":           "Medium",
        "decision":            "Approve",
        "threshold_used":      0.167,
    }
    predictor.threshold = 0.167
    predictor.model_path = "/mock/credit_model.pkl"
    return predictor


@pytest.fixture
def client_mocked(mock_predictor):
    """
    TestClient with the model predictor fully mocked.
    No disk access or model file required.
    """
    with patch("app.get_predictor", return_value=mock_predictor):
        from app import app
        with TestClient(app) as client:
            yield client


# ---------------------------------------------------------------------------
# 1. Root and health endpoints
# ---------------------------------------------------------------------------
class TestHealthEndpoints:

    def test_root_returns_200(self, client_mocked):
        response = client_mocked.get("/")
        assert response.status_code == 200

    def test_root_contains_service_name(self, client_mocked):
        data = client_mocked.get("/").json()
        assert "service" in data
        assert "Credit Risk" in data["service"]

    def test_root_has_docs_link(self, client_mocked):
        data = client_mocked.get("/").json()
        assert "docs" in data

    def test_health_returns_200(self, client_mocked):
        response = client_mocked.get("/health")
        assert response.status_code == 200

    def test_health_schema(self, client_mocked):
        data = client_mocked.get("/health").json()
        required_keys = ["status", "model_loaded", "model_path", "threshold", "timestamp"]
        for key in required_keys:
            assert key in data, f"Missing key in /health response: {key}"

    def test_model_info_returns_200(self, client_mocked):
        response = client_mocked.get("/model/info")
        assert response.status_code == 200

    def test_model_info_contains_regulatory(self, client_mocked):
        data = client_mocked.get("/model/info").json()
        assert "regulatory_context" in data


# ---------------------------------------------------------------------------
# 2. /predict endpoint — valid inputs
# ---------------------------------------------------------------------------
class TestPredictEndpoint:

    def test_valid_request_returns_200(self, client_mocked):
        response = client_mocked.post("/predict", json=VALID_APPLICATION)
        assert response.status_code == 200

    def test_response_has_required_fields(self, client_mocked):
        data = client_mocked.post("/predict", json=VALID_APPLICATION).json()
        required = [
            "prediction", "probability_approve", "probability_default",
            "risk_tier", "decision", "threshold_used", "model_version", "timestamp",
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_prediction_is_binary(self, client_mocked):
        data = client_mocked.post("/predict", json=VALID_APPLICATION).json()
        assert data["prediction"] in (0, 1)

    def test_probabilities_sum_to_one(self, client_mocked):
        data = client_mocked.post("/predict", json=VALID_APPLICATION).json()
        total = data["probability_approve"] + data["probability_default"]
        assert abs(total - 1.0) < 1e-5, f"Probabilities should sum to 1, got {total}"

    def test_probabilities_in_unit_interval(self, client_mocked):
        data = client_mocked.post("/predict", json=VALID_APPLICATION).json()
        assert 0.0 <= data["probability_approve"] <= 1.0
        assert 0.0 <= data["probability_default"] <= 1.0

    def test_risk_tier_is_valid(self, client_mocked):
        data = client_mocked.post("/predict", json=VALID_APPLICATION).json()
        assert data["risk_tier"] in ("Low", "Medium", "High")

    def test_decision_matches_prediction(self, client_mocked):
        data = client_mocked.post("/predict", json=VALID_APPLICATION).json()
        if data["prediction"] == 1:
            assert data["decision"] == "Approve"
        else:
            assert data["decision"] == "Decline"

    def test_model_version_present(self, client_mocked):
        data = client_mocked.post("/predict", json=VALID_APPLICATION).json()
        assert data["model_version"] == "1.0.0"


# ---------------------------------------------------------------------------
# 3. /predict endpoint — invalid inputs (Pydantic validation)
# ---------------------------------------------------------------------------
class TestPredictValidation:

    def test_missing_field_returns_422(self, client_mocked):
        """Required field missing -> HTTP 422 Unprocessable Entity."""
        incomplete = {k: v for k, v in VALID_APPLICATION.items() if k != "CreditScore"}
        response = client_mocked.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_invalid_credit_score_too_low(self, client_mocked):
        """CreditScore < 300 should fail validation."""
        bad = {**VALID_APPLICATION, "CreditScore": 100.0}
        response = client_mocked.post("/predict", json=bad)
        assert response.status_code == 422

    def test_invalid_credit_score_too_high(self, client_mocked):
        """CreditScore > 850 should fail validation."""
        bad = {**VALID_APPLICATION, "CreditScore": 900.0}
        response = client_mocked.post("/predict", json=bad)
        assert response.status_code == 422

    def test_negative_income_returns_422(self, client_mocked):
        bad = {**VALID_APPLICATION, "Income": -1.0}
        response = client_mocked.post("/predict", json=bad)
        assert response.status_code == 422

    def test_zero_loan_amount_returns_422(self, client_mocked):
        bad = {**VALID_APPLICATION, "LoanAmount": 0.0}
        response = client_mocked.post("/predict", json=bad)
        assert response.status_code == 422

    def test_invalid_employment_type(self, client_mocked):
        """EmploymentType must be one of the four allowed values."""
        bad = {**VALID_APPLICATION, "EmploymentType": "Contract"}
        response = client_mocked.post("/predict", json=bad)
        assert response.status_code == 422

    def test_underage_applicant(self, client_mocked):
        """Age < 18 should fail validation."""
        bad = {**VALID_APPLICATION, "Age": 17.0}
        response = client_mocked.post("/predict", json=bad)
        assert response.status_code == 422

    def test_empty_body_returns_422(self, client_mocked):
        response = client_mocked.post("/predict", json={})
        assert response.status_code == 422

    def test_loan_exceeds_income_ratio(self, client_mocked):
        """LoanAmount > 15x Income should fail the custom validator."""
        bad = {**VALID_APPLICATION, "Income": 1000.0, "LoanAmount": 20000.0}
        response = client_mocked.post("/predict", json=bad)
        assert response.status_code == 422

    def test_all_employment_types_accepted(self, client_mocked):
        """All four valid EmploymentType values should produce 200."""
        for emp_type in ["Full-time", "Part-time", "Self-Employed", "Unemployed"]:
            payload = {**VALID_APPLICATION, "EmploymentType": emp_type}
            response = client_mocked.post("/predict", json=payload)
            assert response.status_code == 200, f"Failed for EmploymentType={emp_type}"


# ---------------------------------------------------------------------------
# 4. /predict/batch endpoint
# ---------------------------------------------------------------------------
class TestBatchPredictEndpoint:

    def test_batch_returns_200(self, client_mocked):
        payload = [VALID_APPLICATION, HIGH_RISK_APPLICATION]
        response = client_mocked.post("/predict/batch", json=payload)
        assert response.status_code == 200

    def test_batch_response_count(self, client_mocked):
        payload = [VALID_APPLICATION, VALID_APPLICATION, VALID_APPLICATION]
        data = client_mocked.post("/predict/batch", json=payload).json()
        assert data["total"] == 3
        assert len(data["predictions"]) == 3

    def test_batch_approved_declined_sum(self, client_mocked):
        payload = [VALID_APPLICATION, HIGH_RISK_APPLICATION]
        data = client_mocked.post("/predict/batch", json=payload).json()
        assert data["approved"] + data["declined"] == data["total"]

    def test_batch_exceeds_limit_returns_400(self, client_mocked):
        """Batch size > 1000 should return HTTP 400."""
        large_batch = [VALID_APPLICATION] * 1001
        response = client_mocked.post("/predict/batch", json=large_batch)
        assert response.status_code == 400

    def test_batch_single_item(self, client_mocked):
        data = client_mocked.post("/predict/batch", json=[VALID_APPLICATION]).json()
        assert data["total"] == 1

    def test_batch_invalid_item_returns_422(self, client_mocked):
        """Batch with one invalid item should fail the whole request."""
        bad = [{**VALID_APPLICATION, "CreditScore": 9999}]
        response = client_mocked.post("/predict/batch", json=bad)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# 5. Content-type and headers
# ---------------------------------------------------------------------------
class TestContentType:

    def test_predict_requires_json(self, client_mocked):
        """Sending non-JSON should result in a 422 error."""
        response = client_mocked.post(
            "/predict",
            data="not json",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code in (400, 415, 422)

    def test_response_content_type_is_json(self, client_mocked):
        response = client_mocked.post("/predict", json=VALID_APPLICATION)
        assert "application/json" in response.headers["content-type"]
