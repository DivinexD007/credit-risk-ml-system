"""
predict.py
----------
Inference module for the credit risk prediction system.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib

DEFAULT_THRESHOLD = 0.167

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "credit_model.pkl"
)

logger = logging.getLogger(__name__)


class CreditRiskPredictor:

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.model_path = os.path.abspath(model_path)
        self.threshold = threshold
        self.pipeline = self._load_model()

        # Expected columns from the training dataset
        self.expected_columns = [
            "Age",
            "Income",
            "LoanAmount",
            "CreditScore",
            "MonthsEmployed",
            "YearsExperience",
            "NumCreditLines",
            "InterestRate",
            "LoanTerm",
            "EmploymentType",
            "Gender",
            "Education",
            "City",
        ]

    def _load_model(self):

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )

        logger.info(f"Loading model from: {self.model_path}")
        return joblib.load(self.model_path)

    def _dict_to_dataframe(self, input_data: dict) -> pd.DataFrame:
        """
        Convert input dictionary to dataframe and guarantee required columns.
        """

        df = pd.DataFrame([input_data])

        # Fill missing features expected by the pipeline
        defaults = {
            "YearsExperience": 0,
            "Gender": "Male",
            "Education": "Bachelor",
            "City": "Unknown",
        }

        for col in self.expected_columns:
            if col not in df.columns:
                df[col] = defaults.get(col, 0)

        # Ensure correct column order
        df = df[self.expected_columns]

        return df

    def predict(self, input_data: dict) -> dict:

        X = self._dict_to_dataframe(input_data)

        prob_approve = float(self.pipeline.predict_proba(X)[0, 1])
        prob_default = 1.0 - prob_approve
        decision_int = int(prob_approve >= self.threshold)

        if prob_default < 0.15:
            risk_tier = "Low"
        elif prob_default < 0.35:
            risk_tier = "Medium"
        else:
            risk_tier = "High"

        result = {
            "prediction": decision_int,
            "probability_approve": round(prob_approve, 6),
            "probability_default": round(prob_default, 6),
            "risk_tier": risk_tier,
            "decision": "Approve" if decision_int == 1 else "Decline",
            "threshold_used": self.threshold,
        }

        logger.info(
            f"Prediction: decision={result['decision']} "
            f"pd={prob_default:.4f} tier={risk_tier}"
        )

        return result

    def predict_batch(self, records: list[dict]) -> list[dict]:
        return [self.predict(record) for record in records]

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:

        for col in self.expected_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[self.expected_columns]

        probas = self.pipeline.predict_proba(df)[:, 1]
        predictions = (probas >= self.threshold).astype(int)

        results = df.copy()
        results["probability_approve"] = probas.round(6)
        results["probability_default"] = (1 - probas).round(6)
        results["prediction"] = predictions
        results["decision"] = np.where(predictions == 1, "Approve", "Decline")

        results["risk_tier"] = pd.cut(
            1 - probas,
            bins=[-np.inf, 0.15, 0.35, np.inf],
            labels=["Low", "Medium", "High"],
        ).astype(str)

        return results


def score_applicant(
    input_data: dict,
    model_path: str = DEFAULT_MODEL_PATH,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:

    predictor = CreditRiskPredictor(model_path=model_path, threshold=threshold)
    return predictor.predict(input_data)


if __name__ == "__main__":

    example = {
        "Age": 35,
        "Income": 55000,
        "LoanAmount": 15000,
        "CreditScore": 720,
        "MonthsEmployed": 36,
        "NumCreditLines": 3,
        "InterestRate": 7.5,
        "LoanTerm": 36,
        "EmploymentType": "Full-time"
    }

    predictor = CreditRiskPredictor()
    result = predictor.predict(example)

    print(result)