"""Prometheus exposition-format export of a metric bundle.

Renders the metric bundle from :func:`autofm.metrics.compute_all` into the
Prometheus text exposition format, so the pipeline's run can be scraped into
Prometheus + Grafana like any other service KPI.

Batch-friendly by design: ``write_prometheus`` writes a ``.prom`` file that the
node_exporter *textfile collector* picks up - no long-running server needed,
which suits a run-and-exit training/eval job. (For a live serving process you
would instead expose an HTTP ``/metrics`` endpoint with ``prometheus_client``;
this module deliberately has no such dependency.)

Metric naming follows Prometheus conventions: a common ``autofm_`` prefix,
lower_snake_case names, a ``gauge`` type (these are point-in-time evaluation
values), and run context carried as labels (compressor, head, horizon, ...).
"""

from __future__ import annotations

_PREFIX = "autofm"

# (bundle path, metric suffix, HELP text). Flat scalars pulled from the bundle.
_SCALAR_METRICS = [
    (("multiclass", "macro_f1"), "macro_f1", "Macro-averaged F1 over fault classes"),
    (("multiclass", "weighted_f1"), "weighted_f1", "Support-weighted F1 over fault classes"),
    (("multiclass", "balanced_accuracy"), "balanced_accuracy", "Balanced accuracy (multi-class)"),
    (("binary_fault", "precision"), "fault_precision", "Precision of fault-vs-Normal detection"),
    (("binary_fault", "recall"), "fault_recall", "Recall of fault-vs-Normal detection (faults caught)"),
    (("binary_fault", "f1"), "fault_f1", "F1 of fault-vs-Normal detection"),
    (("binary_fault", "pr_auc"), "fault_pr_auc", "PR-AUC (average precision) of fault detection"),
    (("binary_fault", "roc_auc"), "fault_roc_auc", "ROC-AUC of fault detection"),
    (("binary_fault", "brier"), "fault_brier", "Brier score of the fault score (lower is better)"),
    (("binary_fault", "positive_rate_true"), "fault_base_rate", "True fault base rate in the eval set"),
    (("mean_uncertainty",), "mean_uncertainty", "Mean predictive uncertainty / confidence-gap"),
]


def _dig(bundle: dict, path: tuple[str, ...]):
    """Follow a key path into the nested bundle; None if any key is missing."""
    cur = bundle
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _escape_label_value(value: str) -> str:
    """Escape a Prometheus label value (backslash, double-quote, newline)."""
    return (str(value).replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n"))


def _format_labels(labels: dict[str, str] | None) -> str:
    """Render a label dict as ``{k="v",...}`` (empty string when no labels)."""
    if not labels:
        return ""
    inner = ",".join(
        f'{k}="{_escape_label_value(v)}"'
        for k, v in sorted(labels.items())
        if v is not None
    )
    return "{" + inner + "}" if inner else ""


def to_prometheus(
    metrics: dict,
    labels: dict[str, str] | None = None,
    *,
    prefix: str = _PREFIX,
) -> str:
    """Render a metric bundle as Prometheus exposition text.

    Args:
        metrics: the bundle from :func:`autofm.metrics.compute_all`.
        labels:  run context attached to every series (e.g. compressor, head).
        prefix:  metric name prefix (default ``autofm``).

    Each scalar present in the bundle becomes a ``<prefix>_<name>`` gauge with
    a HELP/TYPE header. Metrics absent from the bundle (e.g. AUCs when only one
    class is present) are skipped rather than emitted as a misleading 0.
    """
    label_str = _format_labels(labels)
    lines: list[str] = []

    # n_samples is an informational gauge that frames the rest.
    n_samples = metrics.get("n_samples")
    if n_samples is not None:
        name = f"{prefix}_eval_samples"
        lines.append(f"# HELP {name} Number of samples the metrics were computed on")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name}{label_str} {int(n_samples)}")

    for path, suffix, help_text in _SCALAR_METRICS:
        value = _dig(metrics, path)
        if value is None:
            continue
        name = f"{prefix}_{suffix}"
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name}{label_str} {float(value)}")

    # Per-class F1 as a labelled series so a single panel can break it down.
    per_class = _dig(metrics, ("multiclass", "per_class")) or {}
    if per_class:
        name = f"{prefix}_class_f1"
        lines.append(f"# HELP {name} Per-class F1 over fault classes")
        lines.append(f"# TYPE {name} gauge")
        for cls_name, cls_metrics in per_class.items():
            cls_labels = dict(labels or {})
            cls_labels["fault_class"] = cls_name
            lines.append(f"{name}{_format_labels(cls_labels)} {float(cls_metrics['f1'])}")

    # Trailing newline: the textfile collector expects a complete final line.
    return "\n".join(lines) + "\n"


def write_prometheus(
    path: str,
    metrics: dict,
    labels: dict[str, str] | None = None,
    *,
    prefix: str = _PREFIX,
) -> None:
    """Write the Prometheus exposition text to ``path`` (atomic replace).

    Writes to a temporary file in the same directory and renames it into place,
    so a node_exporter textfile-collector scrape never sees a half-written file.
    """
    import os
    import tempfile
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = to_prometheus(metrics, labels, prefix=prefix)

    fd, tmp = tempfile.mkstemp(dir=str(out.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, out)   # atomic on POSIX
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
