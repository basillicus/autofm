"""Evaluation metrics for the autofm FM pipeline.

Public API (import from ``autofm.metrics`` directly)::

    from autofm.metrics import compute_all, format_summary

Submodules:

* :mod:`autofm.metrics.classification` - the metric computations (multi-class
  fault-type metrics + binary fault-detection metrics incl. ROC/PR-AUC).
* :mod:`autofm.metrics.report` - human-readable rendering of a metric bundle.
* :mod:`autofm.metrics.prometheus` - Prometheus exposition-format export
  (``.prom`` textfile for the node_exporter textfile collector -> Grafana).
"""

from autofm.metrics.classification import (
    NORMAL_CLASS_ID,
    binary_fault_metrics,
    compute_all,
    multiclass_metrics,
)
from autofm.metrics.prometheus import to_prometheus, write_prometheus
from autofm.metrics.report import format_summary

__all__ = [
    "NORMAL_CLASS_ID",
    "binary_fault_metrics",
    "compute_all",
    "multiclass_metrics",
    "format_summary",
    "to_prometheus",
    "write_prometheus",
]
