"""
preprocessing.py
----------------
Builds the feature preprocessing pipeline used by all models in the
credit risk system.

The pipeline follows two core production principles:
  1. No data leakage  - all scalers/encoders are fit on training data only
                        and applied (transform-only) to test/scoring data.
  2. Serialisability  - the fitted ColumnTransformer is saved inside the
                        full sklearn Pipeline, so inference uses the exact
                        same transformations without extra steps.

Feature types
-------------
Numerical : StandardScaler (zero mean, unit variance)
Categorical: OneHotEncoder  (drops first level to avoid dummy-variable trap,
                             ignores categories unseen during training)
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Column definitions
# These match the raw CSV feature set exactly.
# Update here if the dataset schema changes; no other file needs editing.
# ---------------------------------------------------------------------------
NUMERICAL_FEATURES = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "YearsExperience",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Education",
    "City",
    "EmploymentType",
]

TARGET_COLUMN = "LoanApproved"


def build_preprocessor() -> ColumnTransformer:
    """
    Construct and return a ColumnTransformer that:
      - Scales numerical features with StandardScaler
      - One-hot encodes categorical features (drop first to avoid collinearity)

    The transformer is unfitted; call .fit_transform(X_train) or embed it
    inside a sklearn Pipeline before use.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessing transformer.
    """
    numerical_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",   # silently zero-encode unseen categories at inference
        drop="first",              # avoid dummy variable trap
        sparse_output=False,       # return dense array for compatibility
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",          # drop any columns not listed above
    )

    return preprocessor


def build_pipeline(classifier) -> Pipeline:
    """
    Wrap a preprocessor and a classifier into a single sklearn Pipeline.

    Using a Pipeline guarantees:
      - The preprocessor is fitted only on training data during CV
      - Inference calls only require pipeline.predict_proba(X_raw)

    Parameters
    ----------
    classifier : sklearn estimator
        Any sklearn-compatible classifier (LogisticRegression,
        RandomForestClassifier, etc.).

    Returns
    -------
    Pipeline
        Unfitted end-to-end pipeline (preprocessor -> classifier).

    Example
    -------
    >>> from sklearn.linear_model import LogisticRegression
    >>> pipe = build_pipeline(LogisticRegression())
    >>> pipe.fit(X_train, y_train)
    """
    preprocessor = build_preprocessor()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline


def get_feature_names_after_transform(fitted_pipeline: Pipeline) -> list:
    """
    Recover the full ordered list of feature names after one-hot encoding.

    Useful for SHAP explanations and coefficient inspection.

    Parameters
    ----------
    fitted_pipeline : Pipeline
        A fitted pipeline containing a ColumnTransformer as its first step.

    Returns
    -------
    list of str
        Feature names: numerical features first, then OHE-expanded
        categorical features.
    """
    ct = fitted_pipeline.named_steps["preprocessor"]
    ohe = ct.named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    return NUMERICAL_FEATURES + cat_feature_names