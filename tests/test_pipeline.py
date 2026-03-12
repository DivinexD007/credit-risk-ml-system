"""
tests/test_pipeline.py
----------------------
Unit tests for the credit risk ML pipeline components.

Run with:
    python -m pytest tests/test_pipeline.py -v
"""

import os
import sys
import tempfile

import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "monitoring"))

from preprocessing import (
    build_preprocessor,
    build_pipeline,
    get_feature_names_after_transform,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)
from evaluate import (
    compute_ks_statistic,
    compute_business_cost,
    evaluate_model,
)
from drift_monitor import compute_psi, categorise_psi, DriftMonitor


# ---------------------------------------------------------------------------
# Fixtures — matching YOUR dataset columns
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_df():
    """
    Synthetic dataset matching actual schema:
    Age, Income, LoanAmount, CreditScore, YearsExperience,
    Gender, Education, City, EmploymentType
    """
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "Age":             np.random.uniform(20, 70, n),
        "Income":          np.random.uniform(20000, 120000, n),
        "LoanAmount":      np.random.uniform(1000, 50000, n),
        "CreditScore":     np.random.uniform(300, 850, n),
        "YearsExperience": np.random.uniform(0, 40, n),
        "Gender":          np.random.choice(["Male", "Female"], n),
        "Education":       np.random.choice(
            ["High School", "Bachelor's", "Master's", "PhD"], n
        ),
        "City":            np.random.choice(
            ["New York", "Los Angeles", "Chicago", "Houston"], n
        ),
        "EmploymentType":  np.random.choice(
            ["Full-time", "Part-time", "Self-Employed", "Unemployed"], n
        ),
    })


@pytest.fixture
def sample_target(sample_df):
    np.random.seed(42)
    return pd.Series(
        np.random.choice([0, 1], size=len(sample_df), p=[0.77, 0.23]),
        name=TARGET_COLUMN,
    )


@pytest.fixture
def fitted_pipeline(sample_df, sample_target):
    from sklearn.linear_model import LogisticRegression
    pipeline = build_pipeline(
        LogisticRegression(C=0.5, class_weight="balanced", max_iter=500, random_state=42)
    )
    pipeline.fit(sample_df, sample_target)
    return pipeline


# ---------------------------------------------------------------------------
# 1. Preprocessing tests
# ---------------------------------------------------------------------------
class TestPreprocessor:

    def test_build_preprocessor_returns_column_transformer(self):
        from sklearn.compose import ColumnTransformer
        assert isinstance(build_preprocessor(), ColumnTransformer)

    def test_numerical_features_are_scaled(self, sample_df):
        preprocessor = build_preprocessor()
        X_t = preprocessor.fit_transform(sample_df.iloc[:40])
        num_slice = X_t[:, :len(NUMERICAL_FEATURES)]
        assert abs(num_slice.mean()) < 0.5

    def test_output_shape_correct(self, sample_df):
        preprocessor = build_preprocessor()
        X_t = preprocessor.fit_transform(sample_df)
        assert X_t.shape[1] > len(NUMERICAL_FEATURES)

    def test_no_data_leakage(self, sample_df):
        X_train, X_test = sample_df.iloc[:40], sample_df.iloc[40:]
        pre = build_preprocessor()
        pre.fit(X_train)
        result = pre.transform(X_test)
        assert np.isfinite(result).all()

    def test_handles_unseen_categories(self, sample_df):
        pre = build_preprocessor()
        pre.fit(sample_df)
        new_data = sample_df.copy()
        new_data["EmploymentType"] = "Contract"
        result = pre.transform(new_data)
        assert result is not None
        assert np.isfinite(result).all()


class TestPipeline:

    def test_build_pipeline_structure(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        pipeline = build_pipeline(LogisticRegression())
        assert isinstance(pipeline, Pipeline)
        assert "preprocessor" in pipeline.named_steps
        assert "classifier"   in pipeline.named_steps

    def test_pipeline_fit_predict(self, sample_df, sample_target):
        from sklearn.linear_model import LogisticRegression
        pipeline = build_pipeline(
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=200, random_state=42)
        )
        pipeline.fit(sample_df, sample_target)
        predictions = pipeline.predict(sample_df)
        assert len(predictions) == len(sample_df)
        assert set(predictions).issubset({0, 1})

    def test_pipeline_predict_proba(self, sample_df, sample_target):
        from sklearn.linear_model import LogisticRegression
        pipeline = build_pipeline(
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=200, random_state=42)
        )
        pipeline.fit(sample_df, sample_target)
        probas = pipeline.predict_proba(sample_df)
        assert probas.shape == (len(sample_df), 2)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)
        assert (probas >= 0).all() and (probas <= 1).all()

    def test_get_feature_names(self, fitted_pipeline):
        names = get_feature_names_after_transform(fitted_pipeline)
        assert isinstance(names, list)
        assert len(names) >= len(NUMERICAL_FEATURES)
        for feat in NUMERICAL_FEATURES:
            assert feat in names


# ---------------------------------------------------------------------------
# 2. Evaluation tests
# ---------------------------------------------------------------------------
class TestEvaluate:

    @pytest.fixture
    def binary_arrays(self):
        np.random.seed(0)
        y_true  = np.random.choice([0, 1], size=200, p=[0.77, 0.23])
        y_proba = np.clip(y_true + np.random.normal(0, 0.3, 200), 0, 1)
        return y_true, y_proba

    def test_ks_statistic_range(self, binary_arrays):
        y_true, y_proba = binary_arrays
        ks_stat, ks_thresh = compute_ks_statistic(y_true, y_proba)
        assert 0.0 <= ks_stat  <= 1.0
        assert 0.0 <= ks_thresh <= 1.0

    def test_business_cost_structure(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        result = compute_business_cost(y_true, y_pred, cost_fn=5, cost_fp=1)
        assert result["fn_count"]   == 1
        assert result["fp_count"]   == 1
        assert result["fn_cost"]    == 5
        assert result["fp_cost"]    == 1
        assert result["total_cost"] == 6

    def test_evaluate_model_keys(self, fitted_pipeline, sample_df, sample_target):
        metrics = evaluate_model(fitted_pipeline, sample_df, sample_target)
        for key in ["roc_auc", "accuracy", "precision", "recall", "f1",
                    "confusion_matrix", "ks_statistic", "brier_score", "business_cost"]:
            assert key in metrics

    def test_roc_auc_in_valid_range(self, fitted_pipeline, sample_df, sample_target):
        metrics = evaluate_model(fitted_pipeline, sample_df, sample_target)
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_confusion_matrix_sums(self, fitted_pipeline, sample_df, sample_target):
        metrics = evaluate_model(fitted_pipeline, sample_df, sample_target)
        cm    = metrics["confusion_matrix"]
        total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
        assert total == len(sample_target)


# ---------------------------------------------------------------------------
# 3. Drift monitor tests
# ---------------------------------------------------------------------------
class TestDriftMonitor:

    @pytest.fixture
    def reference_series(self):
        np.random.seed(42)
        return np.random.normal(50, 10, 1000)

    @pytest.fixture
    def stable_series(self):
        np.random.seed(1)
        return np.random.normal(50, 10, 500)

    @pytest.fixture
    def drifted_series(self):
        np.random.seed(2)
        return np.random.normal(70, 10, 500)

    def test_psi_stable_is_low(self, reference_series, stable_series):
        psi = compute_psi(reference_series, stable_series)
        assert psi < 0.10, f"Expected PSI < 0.10 for stable data, got {psi:.4f}"

    def test_psi_drifted_is_high(self, reference_series, drifted_series):
        psi = compute_psi(reference_series, drifted_series)
        assert psi > 0.10, f"Expected PSI > 0.10 for drifted data, got {psi:.4f}"

    def test_categorise_psi(self):
        assert categorise_psi(0.05) == "Stable"
        assert categorise_psi(0.15) == "Moderate Drift"
        assert categorise_psi(0.30) == "Significant Drift"

    def test_psi_empty_array(self):
        assert compute_psi(np.array([]), np.array([1, 2, 3])) == 0.0

    def test_psi_zero_variance(self):
        assert compute_psi(np.ones(100), np.ones(50)) == 0.0

    def test_drift_monitor_report(self, sample_df):
        monitor = DriftMonitor(
            reference_data=sample_df.iloc[:40],
            numerical_features=NUMERICAL_FEATURES,
        )
        report = monitor.run_full_report(sample_df.iloc[40:])
        assert report.overall_status in {"OK", "ALERT", "CRITICAL"}
        assert len(report.feature_results) > 0
        assert len(report.recommendations) > 0

    def test_drift_monitor_ks_flag(self, sample_df):
        reference = sample_df.copy()
        current   = sample_df.copy()
        current["Income"] = current["Income"] * 5
        monitor = DriftMonitor(
            reference_data=reference,
            numerical_features=["Income"],
        )
        report = monitor.run_full_report(current)
        income = next(r for r in report.feature_results if r.feature == "Income")
        assert income.psi > 0.10 or income.ks_drift_flag


# ---------------------------------------------------------------------------
# 4. Model serialisation tests
# ---------------------------------------------------------------------------
class TestModelSerialisation:

    def test_joblib_round_trip(self, fitted_pipeline, sample_df):
        import joblib
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        try:
            joblib.dump(fitted_pipeline, tmp_path)
            loaded = joblib.load(tmp_path)
            np.testing.assert_allclose(
                fitted_pipeline.predict_proba(sample_df),
                loaded.predict_proba(sample_df),
                atol=1e-9,
            )
        finally:
            os.unlink(tmp_path)

    def test_model_file_is_nonzero(self, fitted_pipeline):
        import joblib
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        try:
            joblib.dump(fitted_pipeline, tmp_path, compress=3)
            assert os.path.getsize(tmp_path) > 0
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# 5. Train/test split tests
# ---------------------------------------------------------------------------
class TestTrainTestSplit:

    def test_stratified_split_preserves_class_ratio(self, sample_df, sample_target):
        from sklearn.model_selection import train_test_split
        _, _, y_train, y_test = train_test_split(
            sample_df, sample_target,
            test_size=0.2, stratify=sample_target, random_state=42
        )
        assert abs(y_train.mean() - sample_target.mean()) < 0.05
        assert abs(y_test.mean()  - sample_target.mean()) < 0.05

    def test_split_sizes(self, sample_df, sample_target):
        from sklearn.model_selection import train_test_split
        X_train, X_test, _, _ = train_test_split(
            sample_df, sample_target, test_size=0.2, random_state=42
        )
        n = len(sample_df)
        assert len(X_train) == n - int(n * 0.2)
        assert len(X_test)  == int(n * 0.2)
