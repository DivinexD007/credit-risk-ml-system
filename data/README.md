# Dataset

## Loan Risk Prediction Dataset

Place the dataset file `loan_risk_prediction_dataset.csv` in this directory before running training.

### Schema

| Column | Type | Description | Range |
|---|---|---|---|
| `Age` | Float | Applicant age in years | 18–100 |
| `Income` | Float | Annual income (monetary units) | ≥ 0 |
| `LoanAmount` | Float | Requested loan amount | > 0 |
| `CreditScore` | Float | Credit bureau score | 300–850 |
| `MonthsEmployed` | Float | Duration at current employer | ≥ 0 |
| `NumCreditLines` | Int | Number of open credit lines | ≥ 0 |
| `InterestRate` | Float | Loan interest rate (%) | 0–100 |
| `LoanTerm` | Int | Loan duration in months | ≥ 1 |
| `EmploymentType` | String | Employment category | Full-time, Part-time, Self-Employed, Unemployed |
| `LoanApproved` | Int (0/1) | **Target**: 1 = Approved, 0 = Not Approved | — |

### Class Distribution

- **Not Approved (0)**: ~77%
- **Approved (1)**: ~23%
- Total records: 5,000

This imbalance requires stratified train/test splitting and class-weighted models.

### Generating a Synthetic Dataset

If you do not have the original CSV, you can generate a synthetic equivalent:

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000

employment_types = ["Full-time", "Part-time", "Self-Employed", "Unemployed"]
emp_weights      = [0.55, 0.20, 0.15, 0.10]

df = pd.DataFrame({
    "Age":            np.random.uniform(20, 70, n).round(1),
    "Income":         np.random.lognormal(10.8, 0.5, n).round(2),
    "LoanAmount":     np.random.lognormal(9.5, 0.6, n).round(2),
    "CreditScore":    np.clip(np.random.normal(650, 80, n), 300, 850).round(0),
    "MonthsEmployed": np.random.exponential(30, n).round(0),
    "NumCreditLines": np.random.randint(0, 12, n),
    "InterestRate":   np.random.uniform(3, 25, n).round(2),
    "LoanTerm":       np.random.choice([12, 24, 36, 48, 60], n),
    "EmploymentType": np.random.choice(employment_types, n, p=emp_weights),
})

# Synthetic approval logic (credit score + income are primary drivers)
log_odds = (
    -2.5
    + 0.003  * (df["CreditScore"] - 650)
    + 0.000005 * df["Income"]
    - 0.00002  * df["LoanAmount"]
    + np.where(df["EmploymentType"] == "Full-time", 0.4, 0)
    + np.where(df["EmploymentType"] == "Unemployed", -1.2, 0)
    + np.random.normal(0, 0.3, n)
)
p_approve = 1 / (1 + np.exp(-log_odds))
df["LoanApproved"] = (np.random.uniform(0, 1, n) < p_approve).astype(int)

df.to_csv("data/loan_risk_prediction_dataset.csv", index=False)
print(f"Saved {len(df)} rows | Approval rate: {df['LoanApproved'].mean():.3f}")
```
