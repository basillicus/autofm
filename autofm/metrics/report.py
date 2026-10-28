"""Human-readable rendering of a metric bundle.

Kept separate from the metric computation so the numbers (``classification``)
stay plain, testable dicts and the presentation lives in one place.
"""

from __future__ import annotations


def format_summary(metrics: dict, class_names: list[str]) -> str:
    """Render the metric bundle from :func:`compute_all` as a readable block."""
    mc = metrics["multiclass"]
    bf = metrics["binary_fault"]
    lines = [
        "-" * 64,
        f"n_samples            : {metrics['n_samples']}",
        "",
        "Multi-class (fault type)",
        f"  macro_f1           : {mc['macro_f1']:.4f}",
        f"  weighted_f1        : {mc['weighted_f1']:.4f}",
        f"  balanced_accuracy  : {mc['balanced_accuracy']:.4f}",
        "",
        "Binary fault detection (Normal vs any fault)",
        f"  precision          : {bf['precision']:.4f}",
        f"  recall             : {bf['recall']:.4f}  (faults caught)",
        f"  f1                 : {bf['f1']:.4f}",
    ]
    if "pr_auc" in bf:
        lines += [
            f"  pr_auc             : {bf['pr_auc']:.4f}  (headline under imbalance)",
            f"  roc_auc            : {bf['roc_auc']:.4f}",
            f"  brier              : {bf['brier']:.4f}  (lower is better)",
        ]
    lines += [
        f"  fault base rate    : {bf['positive_rate_true']:.4f}  (true positives / all)",
    ]
    if "mean_uncertainty" in metrics:
        lines.append(f"  mean_uncertainty   : {metrics['mean_uncertainty']:.4f}")
    lines.append("-" * 64)
    return "\n".join(lines)
