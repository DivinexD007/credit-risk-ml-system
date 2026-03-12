"""
train_model.py
--------------
End-to-end training script for the credit risk prediction model.

Workflow
--------
1. Load raw dataset from data/
2. Clean and validate input
3. Stratified train/test split  (80/20, preserves 23% approval rate)
4. Build sklearn Pipeline       (preprocessor + LogisticRegression)
5. Hyperparameter optimisation  (RandomizedSearchCV on Random Forest challenger)
6. Evaluate champion + challenger metrics
7. Save champion pipeline to models/credit_model.pkl

Usage
-----
    python src/train_model.py
    python src/train_model.py --data data/loan_risk_prediction_dataset.csv
    python src/train_model.py --model rf   # train Random Forest as champion

Champion model: Logistic Regression
  - Full coefficient interpretability (required for ECOA adverse action notices)
  - Calibrated probabilities for Basel III capital calculations
  - Competitive AUC, higher recall than RF on this dataset

Challenger model: Random Forest (saved separately for shadow-mode comparison)
"""

import argparse
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    brier_score_loss,
)
from scipy.stats import randint

# Project-level imports
from preprocessing import (
    build_pipeline,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)
from evaluate import evaluate_model, print_evaluation_report

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(filepath: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and minimally clean the raw loan dataset.

    Steps
    -----
    - Drop rows with missing target (LoanApproved)
    - Median-impute missing numerical values
    - Return features (X) and target (y) separately

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    X : pd.DataFrame
        Raw feature matrix (no preprocessing applied yet).
    y : pd.Series
        Binary target (1 = Approved, 0 = Not Approved).
    """
    print(f"[DATA] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)

    # Drop rows where the target is missing
    df = df.dropna(subset=[TARGET_COLUMN])

    # Separate features and target
    X = df.drop(TARGET_COLUMN, axis=1).copy()
    y = df[TARGET_COLUMN].astype(int)

    # Median-impute numerical missing values
    for col in NUMERICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())

    print(f"[DATA] Loaded {len(y):,} samples | Approval rate: {y.mean():.3f}")
    print(f"[DATA] Features: {list(X.columns)}")
    return X, y


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> tuple:
    """
    Stratified train/test split.

    Stratification preserves the 23% approval rate in both partitions,
    preventing evaluation bias from random class imbalance.

    Parameters
    ----------
    X, y : features and target
    test_size : float, fraction reserved for test set
    seed : int, random seed for reproducibility

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,           # critical: preserves class ratio
    )

    print(f"[SPLIT] Train: {len(y_train):,} | Test: {len(y_test):,}")
    print(f"[SPLIT] Train approval rate: {y_train.mean():.3f} | "
          f"Test approval rate: {y_test.mean():.3f}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def build_logistic_regression_pipeline() -> object:
    """
    Build the champion Logistic Regression pipeline.

    C=0.5 provides moderate L2 regularisation, selected via 10-fold CV sweep.
    class_weight='balanced' adjusts for the 77/23 class imbalance.
    """
    clf = LogisticRegression(
        C=0.5,
        class_weight="balanced",
        max_iter=1000,
        random_state=SEED,
        solver="lbfgs",
    )
    return build_pipeline(clf)


def build_random_forest_pipeline() -> object:
    """
    Build the Random Forest challenger pipeline (default hyperparameters).
    Full hyperparameter tuning is done via tune_random_forest().
    """
    clf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        oob_score=True,
        random_state=SEED,
        n_jobs=-1,
    )
    return build_pipeline(clf)


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------
def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 30,
) -> object:
    """
    Tune Random Forest hyperparameters using RandomizedSearchCV.

    RandomizedSearchCV samples n_iter combinations from the parameter
    distributions, which is more efficient than exhaustive GridSearchCV
    for large parameter spaces.

    Parameters
    ----------
    X_train, y_train : training data
    n_iter : int, number of parameter combinations to sample

    Returns
    -------
    Best fitted pipeline from the search.
    """
    print(f"\n[TUNING] RandomizedSearchCV | n_iter={n_iter} | 5-fold stratified CV")

    base_pipeline = build_random_forest_pipeline()

    # Parameter search space
    param_dist = {
        "classifier__n_estimators":     randint(100, 500),
        "classifier__max_depth":        [None, 5, 10, 15, 20],
        "classifier__min_samples_leaf": randint(2, 20),
        "classifier__max_features":     ["sqrt", "log2", 0.3, 0.5],
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_strategy,
        scoring="roc_auc",
        refit=True,            # refit best params on full training set
        n_jobs=-1,
        random_state=SEED,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"[TUNING] Best params  : {search.best_params_}")
    print(f"[TUNING] Best CV AUC  : {search.best_score_:.4f}")

    return search.best_estimator_


# ---------------------------------------------------------------------------
# Cross-validation stability check
# ---------------------------------------------------------------------------
def run_cross_validation(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "Model",
    n_splits: int = 5,
) -> dict:
    """
    Run stratified k-fold cross-validation and return summary metrics.

    CV std is the key stability metric: std > 0.04 is flagged as unstable
    by SR 11-7 model validators.

    Parameters
    ----------
    pipeline : fitted or unfitted sklearn Pipeline
    X_train, y_train : training data
    model_name : str, label for display
    n_splits : int, number of CV folds

    Returns
    -------
    dict with keys: mean_auc, std_auc, scores
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

    result = {
        "mean_auc": scores.mean(),
        "std_auc":  scores.std(),
        "scores":   scores.tolist(),
    }

    stability = (
        "Stable"   if scores.std() < 0.02 else
        "Moderate" if scores.std() < 0.04 else
        "UNSTABLE (SR 11-7 flag)"
    )

    print(f"[CV] {model_name:<25} | Mean AUC: {scores.mean():.4f} "
          f"| Std: {scores.std():.4f} | {stability}")
    return result


# ---------------------------------------------------------------------------
# Model saving
# ---------------------------------------------------------------------------
def save_model(pipeline, model_name: str = "credit_model") -> str:
    """
    Save a fitted sklearn Pipeline to the models directory using joblib.

    The full pipeline (preprocessor + model) is serialised so that the
    inference service only needs to call pipeline.predict_proba(X_raw)
    without applying any separate preprocessing.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    model_name : str, base filename (without extension)

    Returns
    -------
    str : full path to saved .pkl file
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    filepath = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    joblib.dump(pipeline, filepath, compress=3)

    size_kb = os.path.getsize(filepath) / 1024
    print(f"[SAVE] Model saved to: {filepath}  ({size_kb:.1f} KB)")
    return filepath


def save_model_card(metrics: dict, params: dict, filepath: str) -> None:
    """
    Save a JSON model card alongside the .pkl for registry / audit trail.

    Parameters
    ----------
    metrics : dict, evaluation metrics (AUC, Brier, KS, etc.)
    params : dict, hyperparameters used
    filepath : str, path to model card JSON
    """
    card = {
        "created_at":     datetime.utcnow().isoformat() + "Z",
        "framework":      "scikit-learn",
        "regulatory":     "SR 11-7 / Basel III IRB / ECOA",
        "metrics":        metrics,
        "hyperparameters": params,
    }
    with open(filepath, "w") as f:
        json.dump(card, f, indent=2)
    print(f"[SAVE] Model card saved to: {filepath}")


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------
def train(data_path: str, champion: str = "lr") -> None:
    """
    Full training pipeline: load -> split -> train -> evaluate -> save.

    Parameters
    ----------
    data_path : str, path to CSV dataset
    champion : str, 'lr' for Logistic Regression, 'rf' for Random Forest
    """
    print("\n" + "=" * 60)
    print("  CREDIT RISK MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load data
    X, y = load_data(data_path)

    # 2. Stratified split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Build and train champion model
    print(f"\n[TRAIN] Champion: {champion.upper()}")
    if champion == "lr":
        champion_pipeline = build_logistic_regression_pipeline()
        champion_name = "LogisticRegression"
        hparams = {"C": 0.5, "class_weight": "balanced", "max_iter": 1000}
    else:
        champion_pipeline = tune_random_forest(X_train, y_train)
        champion_name = "RandomForest (tuned)"
        hparams = champion_pipeline.named_steps["classifier"].get_params()

    champion_pipeline.fit(X_train, y_train)
    print(f"[TRAIN] {champion_name} fitted on {len(y_train):,} samples")

    # 4. Cross-validation stability check
    print("\n[CV] Running 5-fold stratified cross-validation...")
    run_cross_validation(champion_pipeline, X_train, y_train, champion_name)

    # 5. Evaluate on held-out test set
    print("\n[EVAL] Evaluating on test set...")
    metrics = evaluate_model(champion_pipeline, X_test, y_test)
    print_evaluation_report(metrics, champion_name)

    # 6. Also train and save Random Forest challenger
    print("\n[CHALLENGER] Training Random Forest challenger...")
    rf_pipeline = build_random_forest_pipeline()
    rf_pipeline.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf_pipeline, X_test, y_test)
    print_evaluation_report(rf_metrics, "RandomForest (baseline)")
    save_model(rf_pipeline, "challenger_rf_model")

    # 7. Save champion model + model card
    save_model(champion_pipeline, "credit_model")
    card_path = os.path.join(MODEL_DIR, "model_card.json")
    save_model_card(metrics, hparams, card_path)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Champion : {champion_name}")
    print(f"  Test AUC : {metrics['roc_auc']:.4f}")
    print(f"  Model    : {os.path.join(MODEL_DIR, 'credit_model.pkl')}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the credit risk model")
    parser.add_argument(
        "--data",
        default=os.path.join(DATA_DIR, "loan_risk_prediction_dataset.csv"),
        help="Path to the dataset CSV file",
    )
    parser.add_argument(
        "--model",
        choices=["lr", "rf"],
        default="lr",
        help="Champion model type: lr=LogisticRegression, rf=RandomForest",
    )
    args = parser.parse_args()
    train(data_path=args.data, champion=args.model)
