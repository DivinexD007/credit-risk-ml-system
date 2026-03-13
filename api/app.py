"""
api/app.py
----------
FastAPI inference service for the Credit Risk Prediction System.
"""

import os
import sys
import logging
from datetime import datetime, UTC
from typing import Literal, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import CreditRiskPredictor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("credit_risk_api")


MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "credit_model.pkl"),
)

DECISION_THRESHOLD = float(os.environ.get("DECISION_THRESHOLD", "0.167"))

_predictor: Optional[CreditRiskPredictor] = None


def get_predictor() -> CreditRiskPredictor:
    global _predictor

    if _predictor is None:
        logger.info(f"Loading model from {MODEL_PATH}")
        _predictor = CreditRiskPredictor(
            model_path=MODEL_PATH,
            threshold=DECISION_THRESHOLD,
        )
        logger.info("Model loaded")

    return _predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting API service")

    try:
        get_predictor()
        logger.info("Model preloaded successfully")
    except Exception as e:
        logger.warning(f"Model preload failed: {e}")

    yield

    logger.info("Shutting down API service")


app = FastAPI(
    title="Credit Risk Prediction API",
    description="Production ML inference service for loan approval scoring.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoanApplication(BaseModel):

    Age: float = Field(..., ge=18, le=100)
    Income: float = Field(..., ge=0)
    LoanAmount: float = Field(..., gt=0)
    CreditScore: float = Field(..., ge=300, le=850)

    MonthsEmployed: float = Field(..., ge=0)
    NumCreditLines: int = Field(..., ge=0)
    InterestRate: float = Field(..., ge=0, le=100)
    LoanTerm: int = Field(..., ge=1)

    EmploymentType: Literal[
        "Full-time",
        "Part-time",
        "Self-Employed",
        "Unemployed"
    ]

    @model_validator(mode="after")
    def validate_loan_to_income_ratio(self):
        if self.LoanAmount > (self.Income * 15):
            raise ValueError("LoanAmount cannot exceed 15x Income")
        return self


class PredictionResponse(BaseModel):

    prediction: Literal[0, 1]
    probability_approve: float = Field(..., ge=0.0, le=1.0)
    probability_default: float = Field(..., ge=0.0, le=1.0)
    risk_tier: Literal["Low", "Medium", "High"]
    decision: Literal["Approve", "Decline"]
    threshold_used: float
    model_version: str
    timestamp: str


class BatchPredictionResponse(BaseModel):

    predictions: List[PredictionResponse]
    total: int
    approved: int
    declined: int


class HealthResponse(BaseModel):

    status: str
    model_loaded: bool
    model_path: str
    threshold: float
    timestamp: str


@app.middleware("http")
async def log_requests(request: Request, call_next):

    start = datetime.now(UTC)

    response = await call_next(request)

    duration_ms = (datetime.now(UTC) - start).total_seconds() * 1000

    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.2f}ms)"
    )

    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):

    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


@app.get("/", tags=["Health"])
def root():

    return {
        "service": "Credit Risk Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():

    try:
        get_predictor()
        model_loaded = True
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_path=MODEL_PATH,
        threshold=DECISION_THRESHOLD,
        timestamp=datetime.now(UTC).isoformat(),
    )


@app.get("/model/info", tags=["Model"])
def model_info():

    return {
        "model_type": "LogisticRegression (champion)",
        "challenger": "RandomForestClassifier (shadow mode)",
        "framework": "scikit-learn",
        "version": "1.0.0",
        "decision_threshold": DECISION_THRESHOLD,
        "threshold_derivation": "theta* = C_FP/(C_FP+C_FN) = 1/(1+5) = 0.167",
        "regulatory_context": [
            "SR 11-7 (Model Risk Management)",
            "Basel III IRB (PD estimation)",
            "ECOA Reg B (Adverse action notices)",
            "GDPR Article 22 (Automated decision-making)"
        ],
        "features": {
            "numerical": [
                "Age",
                "Income",
                "LoanAmount",
                "CreditScore",
                "MonthsEmployed",
                "NumCreditLines",
                "InterestRate",
                "LoanTerm"
            ],
            "categorical": [
                "EmploymentType"
            ]
        }
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(application: LoanApplication):

    try:
        predictor = get_predictor()

        result = predictor.predict(application.model_dump())

        prediction_value = 1 if int(result.get("prediction", 0)) == 1 else 0
        probability_approve = float(result.get("probability_approve", 0.0))
        probability_approve = max(0.0, min(1.0, probability_approve))
        probability_default = 1.0 - probability_approve

        if probability_default < 0.15:
            risk_tier = "Low"
        elif probability_default < 0.35:
            risk_tier = "Medium"
        else:
            risk_tier = "High"

        return PredictionResponse(
            prediction=prediction_value,
            probability_approve=probability_approve,
            probability_default=probability_default,
            risk_tier=risk_tier,
            decision="Approve" if prediction_value == 1 else "Decline",
            threshold_used=float(result.get("threshold_used", DECISION_THRESHOLD)),
            model_version="1.0.0",
            timestamp=datetime.now(UTC).isoformat(),
        )

    except FileNotFoundError as e:

        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {str(e)}"
        )

    except Exception as e:

        logger.error(f"Prediction error: {e}", exc_info=True)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
def predict_batch(applications: List[LoanApplication]):

    if len(applications) > 1000:

        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds 1000"
        )

    try:

        predictor = get_predictor()

        timestamp = datetime.now(UTC).isoformat()

        results = []

        for application in applications:

            result = predictor.predict(application.model_dump())
            prediction_value = 1 if int(result.get("prediction", 0)) == 1 else 0
            probability_approve = float(result.get("probability_approve", 0.0))
            probability_approve = max(0.0, min(1.0, probability_approve))
            probability_default = 1.0 - probability_approve

            if probability_default < 0.15:
                risk_tier = "Low"
            elif probability_default < 0.35:
                risk_tier = "Medium"
            else:
                risk_tier = "High"

            results.append(
                PredictionResponse(
                    prediction=prediction_value,
                    probability_approve=probability_approve,
                    probability_default=probability_default,
                    risk_tier=risk_tier,
                    decision="Approve" if prediction_value == 1 else "Decline",
                    threshold_used=float(result.get("threshold_used", DECISION_THRESHOLD)),
                    model_version="1.0.0",
                    timestamp=timestamp
                )
            )

        approved = sum(1 for r in results if r.prediction == 1)

        return BatchPredictionResponse(
            predictions=results,
            total=len(results),
            approved=approved,
            declined=len(results) - approved
        )

    except Exception as e:

        logger.error(f"Batch prediction error: {e}", exc_info=True)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
