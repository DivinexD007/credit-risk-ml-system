"""
evaluate.py
-----------
Model evaluation module for credit risk classification.

Implements both standard ML metrics and domain-specific credit risk metrics
used in regulated banking environments (SR 11-7, Basel III IRB).

Standard metrics
----------------
- ROC-AUC        : Primary discrimination metric
- Accuracy        : Reported but NOT used for model selection (misleading on
                    imbalanced data; a trivial "reject all" classifier gets ~77%)
- Precision       : Positive predictive value for the approved class
- Recall          : True positive rate — directly linked to financial loss prevention
- F1 Score        : Harmonic mean of precision and recall
- Confusion Matrix: TP, FP, FN, TN breakdown

Domain-specific metrics
-----------------------
- KS Statistic    : Regulatory standard at Tier-1 banks; maximum separation between
                    the cumulative score distributions of defaulters vs non-defaulters
- Brier Score     : Probabilistic calibration quality; lower is better
- Business Cost   : Weighted error cost with FN=5x, FP=1x (missed default vs
                    false rejection under Basel III asymmetric loss structure)
- Average Precision: Area under the Precision-Recall curve; more informative
                    than AUC for severely imbalanced datasets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


# Business cost weights (Basel III asymmetric loss structure)
# FN = missed default (approve a borrower who defaults) — capital loss
# FP = false rejection (decline a creditworthy applicant) — foregone interest
COST_FN = 5   # 5x more expensive than a false positive
COST_FP = 1


def compute_ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> tuple:
    """
    Compute the Kolmogorov-Smirnov statistic for a credit scorecard.

    KS = max | CDF_defaulters(score) - CDF_non_defaulters(score) |

    Industry benchmarks:
        KS < 0.20  : Poor
        0.20-0.40  : Fair
        0.40-0.60  : Good
        0.60-0.75  : Very Good
        > 0.75     : Exceptional (investigate for overfitting)

    Parameters
    ----------
    y_true  : array-like, true binary labels
    y_proba : array-like, predicted probabilities for class 1

    Returns
    -------
    ks_stat     : float, KS statistic value
    ks_threshold: float, score threshold at which KS is maximised
    """
    thresholds = np.linspace(0, 1, 300)
    tprs, fprs = [], []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        tprs.append(tp / (tp + fn + 1e-9))
        fprs.append(fp / (fp + tn + 1e-9))

    ks_values = np.abs(np.array(tprs) - np.array(fprs))
    ks_stat = ks_values.max()
    ks_threshold = thresholds[ks_values.argmax()]

    return float(ks_stat), float(ks_threshold)


def compute_business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: int = COST_FN,
    cost_fp: int = COST_FP,
) -> dict:
    """
    Compute the total business cost under the asymmetric loss structure.

    Parameters
    ----------
    y_true, y_pred : true labels and predicted labels
    cost_fn : int, cost multiplier for false negatives (missed defaults)
    cost_fp : int, cost multiplier for false positives (false rejections)

    Returns
    -------
    dict with fn_count, fp_count, fn_cost, fp_cost, total_cost
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "fn_count":   int(fn),
        "fp_count":   int(fp),
        "fn_cost":    int(cost_fn * fn),
        "fp_cost":    int(cost_fp * fp),
        "total_cost": int(cost_fn * fn + cost_fp * fp),
    }


def evaluate_model(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """
    Compute the full evaluation metric suite for a fitted pipeline.

    Parameters
    ----------
    pipeline  : fitted sklearn Pipeline with predict_proba method
    X_test    : raw (unprocessed) test features
    y_test    : true binary labels for test set
    threshold : float, decision threshold for converting probabilities to
                binary predictions (default 0.5; use 0.167 for optimal
                business cost under FN=5, FP=1 cost structure)

    Returns
    -------
    dict containing all computed metrics
    """
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    ks_stat, ks_threshold = compute_ks_statistic(y_test.values, y_proba)
    biz_cost = compute_business_cost(y_test.values, y_pred)

    return {
        # Standard classification metrics
        "roc_auc":            float(roc_auc_score(y_test, y_proba)),
        "accuracy":           float(accuracy_score(y_test, y_pred)),
        "precision":          float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":             float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":                 float(f1_score(y_test, y_pred, zero_division=0)),
        "average_precision":  float(average_precision_score(y_test, y_proba)),

        # Confusion matrix components
        "true_positives":   int(tp),
        "false_positives":  int(fp),
        "true_negatives":   int(tn),
        "false_negatives":  int(fn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],

        # Domain-specific credit risk metrics
        "ks_statistic":     ks_stat,
        "ks_threshold":     ks_threshold,
        "brier_score":      float(brier_score_loss(y_test, y_proba)),

        # Business cost analysis (FN=5x, FP=1x)
        "business_cost":    biz_cost,

        # Decision threshold used
        "threshold":        threshold,
    }


def print_evaluation_report(metrics: dict, model_name: str = "Model") -> None:
    """
    Print a formatted evaluation report to stdout.

    Parameters
    ----------
    metrics    : dict, output from evaluate_model()
    model_name : str, label for the report header
    """
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION REPORT: {model_name}")
    print(f"{'=' * 60}")
    print(f"  ROC-AUC             : {metrics['roc_auc']:.4f}")
    print(f"  Average Precision   : {metrics['average_precision']:.4f}")
    print(f"  KS Statistic        : {metrics['ks_statistic']:.4f}  "
          f"(threshold @ {metrics['ks_threshold']:.3f})")
    print(f"  Brier Score         : {metrics['brier_score']:.4f}")
    print()
    print(f"  Accuracy  : {metrics['accuracy']:.4f}  "
          f"(NOT used for selection — misleading on imbalanced data)")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}  "
          f"(TPR — directly linked to financial loss prevention)")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print()
    cm = metrics["confusion_matrix"]
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
    print(f"    FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")
    print()
    bc = metrics["business_cost"]
    print(f"  Business Cost (FN=5x, FP=1x):")
    print(f"    FN cost  : {bc['fn_cost']:,}  ({bc['fn_count']} missed defaults)")
    print(f"    FP cost  : {bc['fp_cost']:,}  ({bc['fp_count']} false rejections)")
    print(f"    Total    : {bc['total_cost']:,}")
    print(f"{'=' * 60}")


def plot_roc_curve(
    models: list,
    y_test: np.ndarray,
    save_path: str = None,
) -> None:
    """
    Plot ROC curves for multiple models on the same axes.

    Parameters
    ----------
    models : list of (name, pipeline) tuples
    y_test : true binary labels
    save_path : str, optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, pipeline in models:
        y_proba = pipeline.predict_proba(y_test)[:, 1] if hasattr(y_test, "columns") else None
        # Accept pre-computed probabilities if pipeline is actually an array
        if hasattr(pipeline, "predict_proba"):
            fpr, tpr, _ = roc_curve(y_test, pipeline.predict_proba(y_test)[:, 1])
        else:
            fpr, tpr, _ = roc_curve(y_test, pipeline)
        auc = roc_auc_score(y_test, pipeline if not hasattr(pipeline, "predict_proba") else
                            pipeline.predict_proba(y_test)[:, 1])
        ax.plot(fpr, tpr, lw=2, label=f"{name}  AUC={auc:.4f}")

    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Credit Risk Models", fontweight="bold")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] ROC curve saved to {save_path}")
    else:
        plt.show()


def fairness_audit(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    ages: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate fairness across age-based demographic groups.

    Implements three fairness criteria:
    1. Statistical Parity  : equal approval rates across groups
    2. Equal Opportunity   : equal TPR (recall) across groups
    3. Disparate Impact    : DI < 0.80 triggers the four-fifths rule

    Parameters
    ----------
    model_name : str
    y_true     : true binary labels
    y_pred     : binary predictions
    y_proba    : predicted probabilities
    ages       : age values for each observation

    Returns
    -------
    pd.DataFrame with per-group fairness metrics
    """
    groups = {
        "Young (<=30)":   ages <= 30,
        "Middle (31-50)": (ages > 30) & (ages <= 50),
        "Older (>50)":    ages > 50,
    }

    rows = []
    for gname, mask in groups.items():
        n = mask.sum()
        if n == 0:
            continue
        yg = np.array(y_true)[mask]
        pg = np.array(y_pred)[mask]
        proba_g = np.array(y_proba)[mask]

        if len(np.unique(yg)) < 2:
            continue

        tn_g, fp_g, fn_g, tp_g = confusion_matrix(yg, pg, labels=[0, 1]).ravel()
        rows.append({
            "Model":    model_name,
            "Group":    gname,
            "N":        int(n),
            "Approval": round(float((pg == 1).mean()), 3),
            "TPR":      round(float(tp_g / (tp_g + fn_g + 1e-9)), 3),
            "FPR":      round(float(fp_g / (fp_g + tn_g + 1e-9)), 3),
            "AUC":      round(float(roc_auc_score(yg, proba_g)), 3),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    ref_approval = df.loc[df.Group == "Middle (31-50)", "Approval"].values
    ref_tpr      = df.loc[df.Group == "Middle (31-50)", "TPR"].values

    if len(ref_approval) > 0:
        df["DI_Ratio"] = (df["Approval"] / ref_approval[0]).round(3)
        df["EOD"]      = (df["TPR"] - ref_tpr[0]).round(3)
        df["DI_Flag"]  = df["DI_Ratio"].apply(lambda x: "FLAG" if x < 0.80 else "OK")

    return df
