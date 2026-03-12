"""
monitoring/drift_monitor.py
---------------------------
Production model monitoring for the Credit Risk Prediction System.

Implements three complementary drift detection methods used by model risk
management teams at Tier-1 financial institutions (SR 11-7, Basel III IRB):

1. Population Stability Index (PSI)
   - Industry standard for quantifying feature and score distribution shift
   - Symmetric KL-divergence between reference (train) and current (score) data
   - Thresholds: < 0.10 stable | 0.10-0.25 moderate | > 0.25 significant

2. Kolmogorov-Smirnov (KS) Two-Sample Test
   - Statistical hypothesis test for distribution equality
   - D = max|CDF_reference(x) - CDF_current(x)|
   - p < 0.05 flags significant distribution shift

3. Score Distribution Monitoring
   - PSI applied to model output scores (predicted PD)
   - Detects model drift before ground-truth labels are available
   - Ground truth typically delayed 30-90 days in credit risk

Usage
-----
    from monitoring.drift_monitor import DriftMonitor

    monitor = DriftMonitor(reference_data=X_train, model=pipeline)
    report = monitor.run_full_report(current_data=X_new)
    monitor.print_report(report)
    monitor.plot_drift_dashboard(report)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import ks_2samp
from datetime import datetime, UTC


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PSI threshold constants (industry standard)
# ---------------------------------------------------------------------------
PSI_STABLE    = 0.10   # below: no action required
PSI_MODERATE  = 0.25   # 0.10-0.25: investigate
# above 0.25: model retraining required

KS_ALPHA      = 0.05   # p-value threshold for KS test


# ---------------------------------------------------------------------------
# Data classes for typed results
# ---------------------------------------------------------------------------
@dataclass
class FeatureDriftResult:
    """Drift analysis result for a single feature."""
    feature:        str
    psi:            float
    psi_category:   str           # Stable | Moderate Drift | Significant Drift
    ks_statistic:   float
    ks_p_value:     float
    ks_drift_flag:  bool          # True if p < KS_ALPHA
    reference_mean: float
    current_mean:   float
    mean_shift_pct: float         # percentage shift in mean


@dataclass
class DriftReport:
    """Full drift monitoring report."""
    timestamp:           str
    n_reference:         int
    n_current:           int
    feature_results:     list[FeatureDriftResult]
    score_psi:           Optional[float]      # PSI on model output scores
    score_psi_category:  Optional[str]
    overall_status:      str                  # OK | ALERT | CRITICAL
    recommendations:     list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core PSI computation
# ---------------------------------------------------------------------------
def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute the Population Stability Index between two distributions.

    PSI = sum( (p_current - p_expected) * ln(p_current / p_expected) )

    Bins are defined by reference (training) quantiles. This is the correct
    approach: we measure how much the current population has shifted relative
    to the population the model was trained on.

    Epsilon clipping avoids log(0) — standard practice in PSI implementations.

    Parameters
    ----------
    reference : np.ndarray
        Feature values from the reference (training) population.
    current : np.ndarray
        Feature values from the current (scoring) population.
    n_bins : int
        Number of quantile-based bins (default 10 = deciles).

    Returns
    -------
    float : PSI value
    """
    reference = np.array(reference).flatten()
    current   = np.array(current).flatten()

    # Remove NaN values
    reference = reference[~np.isnan(reference)]
    current   = current[~np.isnan(current)]

    if len(reference) == 0 or len(current) == 0:
        logger.warning("Empty array passed to compute_psi — returning 0.0")
        return 0.0

    # Build quantile breakpoints from reference distribution only
    quantiles  = np.linspace(0, 100, n_bins + 1)
    bin_edges  = np.unique(np.percentile(reference, quantiles))

    # Guard against zero-variance features (all values identical)
    if len(bin_edges) < 2:
        return 0.0

    # Count observations falling into each bin
    ref_counts  = np.histogram(reference, bins=bin_edges)[0]
    curr_counts = np.histogram(current,   bins=bin_edges)[0]

    # Convert to proportions, clip to avoid log(0)
    ref_pct  = np.clip(ref_counts  / ref_counts.sum(),  1e-6, None)
    curr_pct = np.clip(curr_counts / curr_counts.sum(), 1e-6, None)

    # PSI formula (symmetric KL-divergence variant)
    psi = float(np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct)))
    return psi


def categorise_psi(psi_value: float) -> str:
    """
    Map a PSI value to a human-readable drift category.

    Parameters
    ----------
    psi_value : float

    Returns
    -------
    str : 'Stable' | 'Moderate Drift' | 'Significant Drift'
    """
    if psi_value < PSI_STABLE:
        return "Stable"
    elif psi_value < PSI_MODERATE:
        return "Moderate Drift"
    else:
        return "Significant Drift"


# ---------------------------------------------------------------------------
# Main DriftMonitor class
# ---------------------------------------------------------------------------
class DriftMonitor:
    """
    Full-featured drift monitoring system for production ML.

    Compares a reference dataset (training data) against a current dataset
    (recent scoring data) to detect distributional shift that could degrade
    model performance.

    Parameters
    ----------
    reference_data : pd.DataFrame
        Training dataset (used as the reference population).
    numerical_features : list of str
        Column names of numerical features to monitor.
    model : sklearn Pipeline, optional
        Fitted model pipeline. If provided, score distribution PSI is
        computed using model.predict_proba(). Required for score monitoring.

    Example
    -------
    >>> monitor = DriftMonitor(
    ...     reference_data=X_train,
    ...     numerical_features=["Age", "Income", "CreditScore"],
    ...     model=fitted_pipeline,
    ... )
    >>> report = monitor.run_full_report(X_scoring)
    >>> monitor.print_report(report)
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        numerical_features: Optional[list] = None,
        model=None,
        n_bins: int = 10,
    ):
        self.reference_data     = reference_data
        self.numerical_features = numerical_features or self._infer_numerical(reference_data)
        self.model              = model
        self.n_bins             = n_bins

        # Pre-compute reference scores if model is provided
        self._reference_scores = None
        if model is not None:
            try:
                self._reference_scores = model.predict_proba(reference_data)[:, 1]
                logger.info("Reference scores computed for score PSI monitoring")
            except Exception as e:
                logger.warning(f"Could not compute reference scores: {e}")

    @staticmethod
    def _infer_numerical(df: pd.DataFrame) -> list:
        """Automatically detect numerical columns."""
        return [col for col in df.columns if df[col].dtype in [np.float64, np.int64, float, int]]

    def analyse_feature(
        self,
        feature: str,
        current_data: pd.DataFrame,
    ) -> FeatureDriftResult:
        """
        Run PSI and KS analysis for a single feature.

        Parameters
        ----------
        feature : str
        current_data : pd.DataFrame

        Returns
        -------
        FeatureDriftResult
        """
        ref_values  = self.reference_data[feature].dropna().values
        curr_values = current_data[feature].dropna().values

        # PSI
        psi = compute_psi(ref_values, curr_values, self.n_bins)

        # KS two-sample test
        ks_stat, ks_p = ks_2samp(ref_values, curr_values)

        # Mean shift
        ref_mean  = float(ref_values.mean())
        curr_mean = float(curr_values.mean())
        shift_pct = ((curr_mean - ref_mean) / (abs(ref_mean) + 1e-9)) * 100

        return FeatureDriftResult(
            feature=feature,
            psi=round(psi, 6),
            psi_category=categorise_psi(psi),
            ks_statistic=round(float(ks_stat), 6),
            ks_p_value=round(float(ks_p), 6),
            ks_drift_flag=bool(ks_p < KS_ALPHA),
            reference_mean=round(ref_mean, 4),
            current_mean=round(curr_mean, 4),
            mean_shift_pct=round(shift_pct, 2),
        )

    def run_full_report(self, current_data: pd.DataFrame) -> DriftReport:
        """
        Run the complete drift monitoring suite on the current dataset.

        Parameters
        ----------
        current_data : pd.DataFrame
            Most recent scoring data to compare against the reference.

        Returns
        -------
        DriftReport
        """
        from datetime import datetime

        logger.info(f"Running drift analysis: reference n={len(self.reference_data)}, "
                    f"current n={len(current_data)}")

        # Feature-level drift analysis
        feature_results = []
        for feat in self.numerical_features:
            if feat in current_data.columns:
                result = self.analyse_feature(feat, current_data)
                feature_results.append(result)

        # Score distribution PSI
        score_psi = None
        score_cat = None
        if self.model is not None and self._reference_scores is not None:
            try:
                current_scores = self.model.predict_proba(current_data)[:, 1]
                score_psi = compute_psi(self._reference_scores, current_scores, self.n_bins)
                score_cat = categorise_psi(score_psi)
            except Exception as e:
                logger.warning(f"Score PSI computation failed: {e}")

        # Overall status
        max_psi = max((r.psi for r in feature_results), default=0)
        ks_flags = sum(1 for r in feature_results if r.ks_drift_flag)

        if max_psi >= PSI_MODERATE or (score_psi and score_psi >= PSI_MODERATE):
            overall_status = "CRITICAL"
        elif max_psi >= PSI_STABLE or ks_flags >= 2:
            overall_status = "ALERT"
        else:
            overall_status = "OK"

        # Recommendations
        recommendations = self._generate_recommendations(
            feature_results, score_psi, overall_status
        )

        return DriftReport(
    timestamp=datetime.now(UTC).isoformat(),
    n_reference=len(self.reference_data),
    n_current=len(current_data),
    feature_results=feature_results,
    score_psi=round(score_psi, 6) if score_psi is not None else None,
    score_psi_category=score_cat,
    overall_status=overall_status,
    recommendations=recommendations,
)

    @staticmethod
    def _generate_recommendations(
        feature_results: list,
        score_psi: Optional[float],
        overall_status: str,
    ) -> list:
        """Generate actionable recommendations based on drift findings."""
        recs = []

        drifted_features = [
            r.feature for r in feature_results
            if r.psi_category in ("Moderate Drift", "Significant Drift")
        ]
        if drifted_features:
            recs.append(
                f"Investigate distribution shift in features: {', '.join(drifted_features)}"
            )

        ks_flagged = [r.feature for r in feature_results if r.ks_drift_flag]
        if ks_flagged:
            recs.append(
                f"KS test flagged significant distributional change in: {', '.join(ks_flagged)}"
            )

        if score_psi and score_psi >= PSI_MODERATE:
            recs.append(
                "Score distribution PSI is critical — model retraining required. "
                "Notify model risk management immediately."
            )
        elif score_psi and score_psi >= PSI_STABLE:
            recs.append(
                "Score distribution shows moderate drift — increase monitoring frequency "
                "and prepare retraining pipeline."
            )

        if overall_status == "OK":
            recs.append(
                "No significant drift detected — continue normal monthly monitoring cadence."
            )

        return recs

    def print_report(self, report: DriftReport) -> None:
        """Pretty-print the drift monitoring report to stdout."""
        print(f"\n{'=' * 70}")
        print(f"  DRIFT MONITORING REPORT")
        print(f"  Timestamp : {report.timestamp}")
        print(f"  Reference : {report.n_reference:,} samples")
        print(f"  Current   : {report.n_current:,} samples")
        print(f"  Status    : {report.overall_status}")
        print(f"{'=' * 70}")

        print(f"\n{'FEATURE':<18} {'PSI':>8} {'CATEGORY':<20} {'KS Stat':>8} "
              f"{'p-value':>9} {'Drift?':>7} {'Mean Shift':>11}")
        print("-" * 85)
        for r in sorted(report.feature_results, key=lambda x: x.psi, reverse=True):
            flag = "YES" if r.ks_drift_flag else "NO"
            print(
                f"{r.feature:<18} {r.psi:>8.4f} {r.psi_category:<20} "
                f"{r.ks_statistic:>8.4f} {r.ks_p_value:>9.4f} {flag:>7} "
                f"{r.mean_shift_pct:>+10.1f}%"
            )

        if report.score_psi is not None:
            print(f"\n  Score Distribution PSI: {report.score_psi:.4f} "
                  f"({report.score_psi_category})")

        print(f"\n  Recommendations:")
        for rec in report.recommendations:
            print(f"    - {rec}")

        print(f"\n  PSI Thresholds: < {PSI_STABLE} Stable | "
              f"{PSI_STABLE}-{PSI_MODERATE} Moderate | > {PSI_MODERATE} Significant")
        print(f"  KS p-value threshold: {KS_ALPHA}")
        print(f"{'=' * 70}\n")

    def plot_drift_dashboard(
        self,
        report: DriftReport,
        current_data: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Render a 2-row dashboard:
          Row 1: PSI bar chart
          Row 2: KDE overlays for the top-3 most drifted features

        Parameters
        ----------
        report : DriftReport from run_full_report()
        current_data : pd.DataFrame
        save_path : str, optional path to save the figure
        """
        import seaborn as sns
        sns.set_theme(style="whitegrid", font_scale=1.0)

        colour_map = {
            "Stable":            "#16A34A",
            "Moderate Drift":    "#D97706",
            "Significant Drift": "#DC2626",
        }

        sorted_results = sorted(report.feature_results, key=lambda r: r.psi, reverse=True)

        # ── Figure layout ────────────────────────────────────────────────────
        fig = plt.figure(figsize=(18, 10))
        gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

        # ── Row 1: PSI bar chart (spans all columns) ─────────────────────────
        ax_psi = fig.add_subplot(gs[0, :])
        psi_vals   = [r.psi for r in sorted_results]
        feat_names = [r.feature for r in sorted_results]
        bar_colors = [colour_map[r.psi_category] for r in sorted_results]

        bars = ax_psi.bar(feat_names, psi_vals, color=bar_colors, alpha=0.88,
                          edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, psi_vals):
            ax_psi.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold"
            )
        ax_psi.axhline(PSI_STABLE,   color=colour_map["Moderate Drift"],    ls="--", lw=1.5,
                       label=f"PSI={PSI_STABLE} (Moderate threshold)")
        ax_psi.axhline(PSI_MODERATE, color=colour_map["Significant Drift"], ls="--", lw=1.5,
                       label=f"PSI={PSI_MODERATE} (Significant threshold)")

        legend_patches = [
            mpatches.Patch(color=colour_map["Stable"],            label="Stable (PSI < 0.10)"),
            mpatches.Patch(color=colour_map["Moderate Drift"],    label="Moderate Drift (0.10-0.25)"),
            mpatches.Patch(color=colour_map["Significant Drift"], label="Significant Drift (>= 0.25)"),
        ]
        ax_psi.legend(handles=legend_patches, fontsize=9, loc="upper right")
        ax_psi.set_title(
            f"Population Stability Index (PSI) — Feature Drift Report\n"
            f"Reference: {report.n_reference:,} samples | Current: {report.n_current:,} samples",
            fontweight="bold"
        )
        ax_psi.set_ylabel("PSI")

        # ── Row 2: KDE overlays for top-3 features ───────────────────────────
        top3 = sorted_results[:3]
        for col_idx, r in enumerate(top3):
            ax = fig.add_subplot(gs[1, col_idx])
            ref_vals  = self.reference_data[r.feature].dropna()
            curr_vals = current_data[r.feature].dropna()

            try:
                sns.kdeplot(ref_vals,  ax=ax, color="#2563EB", lw=2,
                            label=f"Reference (n={len(ref_vals):,})", fill=True, alpha=0.15)
                sns.kdeplot(curr_vals, ax=ax, color="#DC2626",  lw=2,
                            label=f"Current (n={len(curr_vals):,})",   fill=True, alpha=0.15)
            except Exception:
                ax.hist(ref_vals,  bins=20, alpha=0.5, color="#2563EB", label="Reference")
                ax.hist(curr_vals, bins=20, alpha=0.5, color="#DC2626",  label="Current")

            title_color = colour_map[r.psi_category]
            flag_txt = "DRIFT" if r.ks_drift_flag else "OK"
            ax.set_title(
                f"{r.feature}\nPSI={r.psi:.4f}  KS={r.ks_statistic:.4f}  [{flag_txt}]",
                fontweight="bold", color=title_color, fontsize=10
            )
            ax.set_xlabel(r.feature)
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)

        fig.suptitle(
            f"Drift Monitoring Dashboard  |  Status: {report.overall_status}  |  {report.timestamp}",
            fontsize=13, fontweight="bold"
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Drift dashboard saved to {save_path}")
        else:
            plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Convenience function: run a quick drift check from the command line
# ---------------------------------------------------------------------------
def run_drift_check(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numerical_features: list,
    model=None,
    plot: bool = True,
    save_path: Optional[str] = None,
) -> DriftReport:
    """
    Run a complete drift check and return the report.

    Parameters
    ----------
    reference_df : pd.DataFrame, training/reference data
    current_df   : pd.DataFrame, current scoring data
    numerical_features : list of str
    model        : fitted sklearn pipeline (optional, for score PSI)
    plot         : bool, whether to render the dashboard
    save_path    : str, optional path to save dashboard image

    Returns
    -------
    DriftReport
    """
    monitor = DriftMonitor(
        reference_data=reference_df,
        numerical_features=numerical_features,
        model=model,
    )
    report = monitor.run_full_report(current_df)
    monitor.print_report(report)

    if plot:
        monitor.plot_drift_dashboard(report, current_df, save_path=save_path)

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    from preprocessing import NUMERICAL_FEATURES

    parser = argparse.ArgumentParser(description="Run drift monitoring report")
    parser.add_argument("--reference", required=True, help="Path to reference CSV (training data)")
    parser.add_argument("--current",   required=True, help="Path to current CSV (scoring data)")
    parser.add_argument("--model",     default=None,  help="Path to .pkl model for score PSI")
    parser.add_argument("--no-plot",   action="store_true", help="Skip dashboard plot")
    parser.add_argument("--save",      default=None,  help="Path to save dashboard image")
    args = parser.parse_args()

    import pandas as pd
    import joblib

    ref_df  = pd.read_csv(args.reference)
    curr_df = pd.read_csv(args.current)
    model   = joblib.load(args.model) if args.model else None

    run_drift_check(
        reference_df=ref_df,
        current_df=curr_df,
        numerical_features=NUMERICAL_FEATURES,
        model=model,
        plot=not args.no_plot,
        save_path=args.save,
    )
