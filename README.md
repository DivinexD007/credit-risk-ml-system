# Credit Risk ML System

**Production-grade machine learning system for loan default prediction in regulated lending environments.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Dataset Description](#3-dataset-description)
4. [Repository Structure](#4-repository-structure)
5. [ML Pipeline](#5-ml-pipeline)
6. [Model Evaluation Metrics](#6-model-evaluation-metrics)
7. [Model Explainability (SHAP)](#7-model-explainability-shap)
8. [Fairness Analysis](#8-fairness-analysis)
9. [Model Monitoring](#9-model-monitoring)
10. [Running the Project Locally](#10-running-the-project-locally)
11. [Docker Deployment](#11-docker-deployment)
12. [API Reference](#12-api-reference)
13. [Architecture](#13-architecture)
14. [Regulatory Compliance](#14-regulatory-compliance)

---

## 1. Project Overview

This repository implements a complete, production-ready **Credit Risk Prediction System** that predicts the probability of loan default for consumer lending applications. The system is built to industry standards used by ML engineering teams at Tier-1 financial institutions.

**Key capabilities:**
- End-to-end sklearn Pipeline (preprocessing + classification) with guaranteed no data leakage
- Champion model: Logistic Regression (full interpretability, calibrated PD estimates)
- Challenger model: Random Forest (higher raw AUC, shadow-mode deployment)
- Real-time inference via FastAPI REST API with Pydantic request validation
- Production drift monitoring: Population Stability Index (PSI) + KS two-sample tests
- SHAP-based per-applicant explainability for regulatory adverse action notices
- Fairness audit across demographic groups (ECOA compliance)
- Docker containerised deployment

---

## 2. Problem Statement

Credit risk assessment determines whether a loan applicant is likely to default. This is a high-stakes binary classification problem with two key characteristics:

### Class Imbalance
The dataset has a 77/23 split (Not Approved / Approved). A naive classifier predicting "reject all" achieves ~77% accuracy while capturing zero defaults. Standard accuracy is therefore economically invalid.

### Asymmetric Business Costs
Under the Basel III capital framework, errors have different financial consequences:

| Error Type | Description | Relative Cost |
|---|---|---|
| **False Negative (FN)** | Approve a borrower who defaults | 5× |
| **False Positive (FP)** | Decline a creditworthy applicant | 1× |

This asymmetry is encoded in the decision threshold:

```
theta* = C_FP / (C_FP + C_FN) = 1 / (1 + 5) = 0.167
```

The model uses threshold **0.167** (not the default 0.5) to minimise total business cost.

### Expected Loss Framework (Basel III IRB)

```
EL = PD × LGD × EAD
```

Where PD (Probability of Default) is the calibrated output of the logistic regression model, LGD = 0.45 (unsecured consumer credit benchmark), and EAD = loan amount.

---

## 3. Dataset Description

**Loan Risk Prediction Dataset** — 5,000 synthetic loan applications.

| Feature | Type | Description |
|---|---|---|
| `Age` | Float | Applicant age (18–100 years) |
| `Income` | Float | Annual income (monetary units) |
| `LoanAmount` | Float | Requested loan amount |
| `CreditScore` | Float | Credit bureau score (300–850) |
| `MonthsEmployed` | Float | Duration at current employer |
| `NumCreditLines` | Int | Number of open credit lines |
| `InterestRate` | Float | Loan interest rate (%) |
| `LoanTerm` | Int | Loan duration in months |
| `EmploymentType` | String | Full-time / Part-time / Self-Employed / Unemployed |
| **`LoanApproved`** | **Int (0/1)** | **Target: 1 = Approved, 0 = Not Approved** |

**Class distribution:** 77% Not Approved | 23% Approved

Place `loan_risk_prediction_dataset.csv` in the `data/` directory. See `data/README.md` for a synthetic data generation script.

---

## 4. Repository Structure

```
credit-risk-ml-system/
│
├── data/
│   ├── loan_risk_prediction_dataset.csv    ← dataset (add manually)
│   └── README.md                           ← dataset schema and generation script
│
├── notebooks/
│   └── credit_risk_analysis.ipynb          ← full exploratory analysis and modelling
│
├── src/
│   ├── preprocessing.py    ← ColumnTransformer pipeline builder
│   ├── train_model.py      ← end-to-end training script (load → split → train → save)
│   ├── evaluate.py         ← ROC-AUC, KS, Brier, confusion matrix, fairness audit
│   └── predict.py          ← CreditRiskPredictor class (load model → predict)
│
├── api/
│   └── app.py              ← FastAPI service (/predict, /predict/batch, /health)
│
├── monitoring/
│   └── drift_monitor.py    ← PSI + KS drift detection, DriftMonitor class
│
├── models/
│   ├── credit_model.pkl            ← champion model (generated by training)
│   ├── challenger_rf_model.pkl     ← challenger (generated by training)
│   └── model_card.json             ← model metadata and metrics
│
├── tests/
│   ├── test_pipeline.py    ← unit tests: preprocessing, evaluation, drift, serialisation
│   └── test_api.py         ← integration tests: all FastAPI endpoints
│
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## 5. ML Pipeline

### Architecture

```
Raw CSV Data
    │
    ▼
Data Loading & Cleaning
├── Drop rows with missing target
└── Median-impute numerical NaN values
    │
    ▼
Stratified Train/Test Split (80/20)
    │
    ▼
sklearn Pipeline
├── ColumnTransformer (Preprocessor)
│   ├── StandardScaler       → numerical features (8 columns)
│   └── OneHotEncoder        → EmploymentType (→ 3 dummy columns, drop first)
│
└── Classifier
    ├── Champion: LogisticRegression(C=0.5, class_weight='balanced')
    └── Challenger: RandomForestClassifier (tuned via RandomizedSearchCV)
    │
    ▼
Hyperparameter Optimisation
└── RandomizedSearchCV (30 iterations, 5-fold stratified CV)
    Parameters: n_estimators, max_depth, min_samples_leaf, max_features
    │
    ▼
Model Evaluation
├── ROC-AUC, Precision, Recall, F1
├── KS Statistic (regulatory standard)
├── Brier Score (calibration quality)
└── Business Cost (FN=5x, FP=1x)
    │
    ▼
joblib Serialisation
└── models/credit_model.pkl (full pipeline: preprocessor + classifier)
```

### Design Decisions

| Decision | Rationale |
|---|---|
| **Logistic Regression as champion** | Full coefficient interpretability (ECOA adverse action), calibrated PD for Basel III, stable CV performance |
| **Pipeline wraps preprocessor + model** | Prevents data leakage during CV; single joblib artefact for inference |
| **Stratified split** | Preserves 23% approval rate in both train/test partitions |
| **`class_weight='balanced'`** | Compensates for 77/23 imbalance without oversampling (which leaks when combined with CV) |
| **Threshold = 0.167** | Optimal under FN=5, FP=1 cost structure; derived from `theta* = C_FP/(C_FP+C_FN)` |
| **RandomizedSearchCV over GridSearchCV** | Better exploration of RF parameter space within fixed computational budget |

---

## 6. Model Evaluation Metrics

### Standard Metrics

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| ROC-AUC | ~0.85 | ~0.87 |
| KS Statistic | ~0.55 | ~0.58 |
| Brier Score | ~0.12 | ~0.11 |
| Recall | Higher | Lower |
| CV Std (stability) | < 0.02 | < 0.02 |

> Exact values depend on your dataset. Run `python src/train_model.py` to see current metrics.

### Why ROC-AUC Alone Is Insufficient

Credit risk practitioners use a suite of domain-specific metrics:

**KS Statistic** — Industry standard for scorecard validation. Measures maximum separation between cumulative score distributions of defaulters vs non-defaulters.

| KS Range | Model Quality |
|---|---|
| < 0.20 | Poor |
| 0.20–0.40 | Fair |
| 0.40–0.60 | Good |
| 0.60–0.75 | Very Good |
| > 0.75 | Exceptional (investigate for overfitting) |

**Brier Score** — Measures probability calibration quality. Miscalibrated probabilities directly cause capital misallocation in the Basel III EL formula.

**Business Cost** — Total cost under FN=5, FP=1 weighting. The only metric that directly quantifies financial impact.

**Average Precision** — Area under the Precision-Recall curve. More informative than AUC for severely imbalanced datasets.

---

## 7. Model Explainability (SHAP)

### Why Explainability Is a Regulatory Requirement

- **ECOA (Equal Credit Opportunity Act)** — Lenders must provide applicants with the specific reasons for adverse credit decisions.
- **GDPR Article 22** — Automated decisions must be explainable to data subjects upon request.
- **SR 11-7** — Model validators must document key drivers of model output to assess conceptual soundness.

### SHAP (SHapley Additive exPlanations)

SHAP provides the theoretically unique attribution of each feature's contribution to an individual prediction:

```
f(x) = phi_0 + sum(phi_j)
```

Where `phi_0` = base rate (expected model output) and `phi_j` = Shapley value of feature j (average marginal contribution across all feature coalitions).

**Why SHAP over alternatives:**

| Property | SHAP | Permutation Importance | LIME |
|---|---|---|---|
| Mathematically unique | ✅ | ❌ | ❌ |
| Per-applicant explanations | ✅ | ❌ | ✅ |
| Global feature ranking | ✅ | ✅ | ❌ |
| Fast for tree models | ✅ (TreeExplainer) | N/A | N/A |

### Running SHAP Analysis

See `notebooks/credit_risk_analysis.ipynb` for full SHAP beeswarm plots, waterfall charts for individual applicants, and global feature importance visualisations.

---

## 8. Fairness Analysis

The system implements three fairness criteria evaluated across age-based demographic groups (Young ≤30, Middle 31–50, Older >50):

### Metrics

**1. Statistical Parity (Approval Rate Equality)**
```
P(Decision=Approve | Group=A) ≈ P(Decision=Approve | Group=B)
```

**2. Equal Opportunity (TPR Equality)**
```
EOD = TPR_protected - TPR_reference
```
Non-zero EOD means creditworthy borrowers in one group are approved at different rates.

**3. Disparate Impact Ratio**
```
DI = Approval_protected / Approval_reference
```
The four-fifths rule flags potential adverse impact when **DI < 0.80**.

### Fairness-Accuracy Trade-off

Imposing equal opportunity constraints reduces AUC because features that legitimately predict risk (e.g. credit history length) are correlated with protected attributes (e.g. age). This trade-off cannot be eliminated without upstream data interventions. Post-processing mitigation (per-group threshold adjustment) is available as an extension.

---

## 9. Model Monitoring

The monitoring system (`monitoring/drift_monitor.py`) detects three failure modes required by SR 11-7:

### Population Stability Index (PSI)

```
PSI = sum( (p_current - p_expected) * ln(p_current / p_expected) )
```

Measures distributional shift between reference (training) and current (scoring) populations.

| PSI Range | Status | Action |
|---|---|---|
| < 0.10 | Stable | Continue normal monthly monitoring |
| 0.10–0.25 | Moderate Drift | Investigate data pipeline changes |
| > 0.25 | Significant Drift | Retrain model, notify model risk management |

### KS Two-Sample Test

Tests whether reference and current feature distributions are drawn from the same underlying distribution. `p < 0.05` flags significant distributional change.

### Score Distribution Monitoring

PSI applied to model output scores (predicted PD). Detects model drift before ground-truth labels are available (credit outcomes are typically delayed 30–90 days).

### Running a Drift Report

```python
from monitoring.drift_monitor import run_drift_check
import pandas as pd, joblib

X_train = pd.read_csv("data/loan_risk_prediction_dataset.csv").sample(4000)
X_new   = pd.read_csv("data/new_scoring_data.csv")
model   = joblib.load("models/credit_model.pkl")

report = run_drift_check(
    reference_df=X_train,
    current_df=X_new,
    numerical_features=["Age", "Income", "LoanAmount", "CreditScore",
                        "MonthsEmployed", "NumCreditLines", "InterestRate", "LoanTerm"],
    model=model,
    plot=True,
)
```

---

## 10. Running the Project Locally

### Prerequisites

- Python 3.11+
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/credit-risk-ml-system.git
cd credit-risk-ml-system

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your dataset
cp /path/to/loan_risk_prediction_dataset.csv data/
```

### Train the Model

```bash
# Train the Logistic Regression champion model (default)
python src/train_model.py

# Train with Random Forest as champion
python src/train_model.py --model rf

# Specify a custom data path
python src/train_model.py --data data/loan_risk_prediction_dataset.csv
```

Training output:
```
[DATA] Loaded 5,000 samples | Approval rate: 0.230
[SPLIT] Train: 4,000 | Test: 1,000
[TRAIN] Champion: LR
[CV] LogisticRegression        | Mean AUC: 0.8512 | Std: 0.0148 | Stable
[EVAL] ROC-AUC: 0.8534  KS: 0.5521  Brier: 0.1187
[SAVE] Model saved to: models/credit_model.pkl (42.3 KB)
```

### Run the API Server

```bash
# Start FastAPI server (requires trained model)
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# The API is now available at:
# http://localhost:8000/docs    <- Swagger UI
# http://localhost:8000/redoc  <- ReDoc UI
```

### Make a Prediction (curl)

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Age": 35,
       "Income": 55000,
       "LoanAmount": 15000,
       "CreditScore": 720,
       "MonthsEmployed": 36,
       "NumCreditLines": 3,
       "InterestRate": 7.5,
       "LoanTerm": 36,
       "EmploymentType": "Full-time"
     }'
```

**Response:**
```json
{
  "prediction": 1,
  "probability_approve": 0.821543,
  "probability_default": 0.178457,
  "risk_tier": "Medium",
  "decision": "Approve",
  "threshold_used": 0.167,
  "model_version": "1.0.0",
  "timestamp": "2025-01-15T10:32:44Z"
}
```

### Make a Prediction (Python)

```python
from src.predict import CreditRiskPredictor

predictor = CreditRiskPredictor()   # loads model once
result = predictor.predict({
    "Age": 35,
    "Income": 55000,
    "LoanAmount": 15000,
    "CreditScore": 720,
    "MonthsEmployed": 36,
    "NumCreditLines": 3,
    "InterestRate": 7.5,
    "LoanTerm": 36,
    "EmploymentType": "Full-time",
})
print(result)
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run only pipeline tests
pytest tests/test_pipeline.py -v

# Run only API tests
pytest tests/test_api.py -v

# Run with coverage report
pytest tests/ --cov=src --cov=api --cov-report=html
```

### Run Drift Monitoring

```bash
# Compare training data vs new scoring data
python monitoring/drift_monitor.py \
    --reference data/loan_risk_prediction_dataset.csv \
    --current   data/new_scoring_data.csv \
    --model     models/credit_model.pkl \
    --save      monitoring/drift_report.png
```

---

## 11. Docker Deployment

### Build and Run

```bash
# 1. Train the model first (must be on disk before building)
python src/train_model.py

# 2. Build the Docker image
docker build -t credit-risk-api .

# 3. Run the container
docker run -p 8000:8000 credit-risk-api

# 4. Test the health endpoint
curl http://localhost:8000/health
```

### Mount a Custom Model

```bash
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e MODEL_PATH=/app/models/credit_model.pkl \
  -e DECISION_THRESHOLD=0.167 \
  credit-risk-api
```

### Docker Compose (with monitoring sidecar)

```yaml
# docker-compose.yml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/credit_model.pkl
      - DECISION_THRESHOLD=0.167
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## 12. API Reference

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Root health check |
| GET | `/health` | Detailed health status |
| GET | `/model/info` | Model metadata and regulatory context |
| POST | `/predict` | Single applicant scoring |
| POST | `/predict/batch` | Batch scoring (max 1,000 records) |

### POST /predict

**Request body:**

```json
{
  "Age": 35.0,
  "Income": 55000.0,
  "LoanAmount": 15000.0,
  "CreditScore": 720.0,
  "MonthsEmployed": 36.0,
  "NumCreditLines": 3,
  "InterestRate": 7.5,
  "LoanTerm": 36,
  "EmploymentType": "Full-time"
}
```

**Response:**

```json
{
  "prediction": 1,
  "probability_approve": 0.821543,
  "probability_default": 0.178457,
  "risk_tier": "Medium",
  "decision": "Approve",
  "threshold_used": 0.167,
  "model_version": "1.0.0",
  "timestamp": "2025-01-15T10:32:44Z"
}
```

**Validation rules:**
- `Age`: 18.0 – 100.0
- `CreditScore`: 300.0 – 850.0
- `LoanAmount` > 0 and ≤ 15× `Income`
- `EmploymentType` ∈ {Full-time, Part-time, Self-Employed, Unemployed}
- All fields required; missing fields return HTTP 422

### POST /predict/batch

Accepts a JSON array of up to 1,000 application objects. Returns:

```json
{
  "predictions": [...],
  "total": 2,
  "approved": 1,
  "declined": 1
}
```

---

## 13. Architecture

```
+----------------------------------------------------------------------+
|                    PRODUCTION ML PIPELINE                            |
+----------------------------------------------------------------------+
|                                                                      |
|  +-----------+   +-------------+   +----------------------------+   |
|  | Data      |-->| Feature     |-->| Model Training             |   |
|  | Layer     |   | Engineering |   | (src/train_model.py)       |   |
|  |           |   |             |   |                            |   |
|  | Loan CSV  |   | Imputation  |   | LogisticRegression (LR)    |   |
|  | Bureau API|   | Scaling     |   | RandomForest (RF)          |   |
|  | Event logs|   | OHE         |   | RandomizedSearchCV tuning  |   |
|  +-----------+   +-------------+   | Cross-validation (5-fold)  |   |
|                                    +------------+---------------+   |
|                                                 |                   |
|  +-----------+   +-------------+   +------------v---------------+   |
|  | Monitoring|<--| Inference   |<--| Model Registry             |   |
|  |           |   | API         |   |                            |   |
|  | PSI alerts|   | (api/app.py)|   | joblib .pkl artefact       |   |
|  | KS tests  |   |             |   | Model card JSON            |   |
|  | Score dist|   | FastAPI     |   | Champion / Challenger       |   |
|  | Fairness  |   | Pydantic    |   | v1.0.0 → v1.1.0 etc.      |   |
|  +-----------+   +-------------+   +----------------------------+   |
+----------------------------------------------------------------------+
```

**Data flow per prediction request:**
```
POST /predict (JSON)
      │
      ▼
Pydantic validation (type + range checks, business rules)
      │
      ▼
CreditRiskPredictor.predict(dict)
      │
      ▼
pd.DataFrame([input_data])  ← preserves column names
      │
      ▼
sklearn Pipeline.predict_proba(X_raw)
├── ColumnTransformer.transform()
│   ├── StandardScaler.transform()  [numerical]
│   └── OneHotEncoder.transform()   [categorical]
└── LogisticRegression.predict_proba()
      │
      ▼
P(Approve) compared to threshold 0.167
      │
      ▼
JSON response {prediction, probability, risk_tier, decision}
      │
      ▼
Audit log entry (for monitoring database)
```

---

## 14. Regulatory Compliance

This system is designed to operate within the following regulatory frameworks:

| Framework | Requirement | Implementation |
|---|---|---|
| **SR 11-7** | Model risk management and validation | Model card, CV stability report, champion/challenger governance |
| **Basel III IRB** | Calibrated PD estimates per rating grade | Logistic regression with Brier score calibration check |
| **ECOA Reg B** | Adverse action reasons (top factors) | SHAP values provide per-applicant feature attribution |
| **GDPR Art. 22** | Explainability of automated decisions | SHAP waterfall charts for individual applicant explanations |
| **Four-Fifths Rule** | Disparate impact threshold DI ≥ 0.80 | Fairness audit in `src/evaluate.py` with DI_Ratio and DI_Flag |

---

## Author

**Tejas Subhash Patil**

---

## License

MIT License — see [LICENSE](LICENSE) for details.
