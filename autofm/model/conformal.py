"""Conformal prediction - calibrated 95% coverage sets.

A classifier gives a best-guess class; conformal prediction turns it into a
**set** of classes guaranteed (under exchangeability) to contain the true label
at a chosen rate, e.g. 95%. The set is calibrated on a held-out slice the model
never trained on, so the coverage is honest rather than assumed.

Backends behind one interface (mirrors the GAT module's pattern):

* **MAPIE path** (preferred): wraps the model's predicted probabilities in a
  tiny prefit sklearn-style shim and calls MAPIE's split-conformal classifier.
* **Fallback path** (no MAPIE, or API mismatch): a direct split-conformal
  score (LAC / "score" method) in numpy - the calibration quantile of
  ``1 - p[true]`` gives a threshold; the prediction set is every class whose
  probability clears it. Same guarantee, no dependency.

The active path is printed at construction. ``calibrate`` + ``predict_set``
return boolean ``(N, K)`` set masks; ``empirical_coverage`` reports the realized
coverage vs the target so the calibration can be checked.
"""

from __future__ import annotations

import numpy as np

try:  # Preferred: MAPIE's conformal machinery.
    from mapie.classification import SplitConformalClassifier
    _HAS_MAPIE = True
except Exception:  # noqa: BLE001 - any import issue falls back to numpy
    _HAS_MAPIE = False


class _PrefitProbaEstimator:
    """Minimal prefit sklearn-style classifier over *precomputed* probabilities.

    MAPIE expects an estimator with ``predict_proba`` / ``classes_`` and sklearn
    tags. We never actually fit a model here - we pass probabilities straight
    through - so MAPIE only does the conformal calibration on top. The estimator
    is indexed by row so MAPIE's internal ``X`` can just be row indices.
    """

    def __init__(self, proba: np.ndarray):
        self._proba = proba
        self.classes_ = np.arange(proba.shape[1])
        # sklearn 1.6+ tag protocol; kept permissive.
        self._estimator_type = "classifier"

    def predict_proba(self, idx) -> np.ndarray:
        rows = np.asarray(idx).reshape(-1).astype(int)
        return self._proba[rows]

    def predict(self, idx) -> np.ndarray:
        return self.predict_proba(idx).argmax(axis=1)

    def get_params(self, deep: bool = True) -> dict:  # noqa: D401 - sklearn api
        return {}

    def __sklearn_is_fitted__(self) -> bool:
        return True


class ConformalPredictor:
    """Split-conformal prediction set wrapper (MAPIE or numpy fallback)."""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.use_mapie = _HAS_MAPIE
        self._threshold: float | None = None     # fallback LAC threshold
        self._mapie = None
        if self.use_mapie:
            print(f"[conformal] using MAPIE split-conformal "
                  f"(target {confidence_level:.0%})")
        else:
            print(f"[conformal] MAPIE not found - using numpy split-conformal "
                  f"LAC (target {confidence_level:.0%})")

    # -- calibration ----------------------------------------------------
    def calibrate(self, proba_cal: np.ndarray, y_cal: np.ndarray) -> "ConformalPredictor":
        """Fit the conformal threshold on the calibration slice.

        Args:
            proba_cal: (N, K) model probabilities on the calibration slice.
            y_cal:     (N,) true labels for that slice.
        """
        proba_cal = np.asarray(proba_cal, dtype=np.float64)
        y_cal = np.asarray(y_cal).astype(int)

        if self.use_mapie:
            try:
                self._calibrate_mapie(proba_cal, y_cal)
                return self
            except Exception as exc:  # noqa: BLE001 - fall back on any API drift
                print(f"[conformal] MAPIE calibration failed ({exc}); "
                      "falling back to numpy LAC")
                self.use_mapie = False

        self._calibrate_numpy(proba_cal, y_cal)
        return self

    def _calibrate_mapie(self, proba_cal: np.ndarray, y_cal: np.ndarray) -> None:
        # Let MAPIE compute the LAC conformity scores on the calibration slice,
        # then derive the inclusion threshold from them with the standard
        # finite-sample quantile. We apply that threshold ourselves at predict
        # time (p >= 1 - thr), which is the LAC set rule - this avoids MAPIE's
        # prefit estimator caching, which otherwise re-scores the wrong rows.
        est = _PrefitProbaEstimator(proba_cal)
        idx = np.arange(len(proba_cal)).reshape(-1, 1)
        self._mapie = SplitConformalClassifier(
            estimator=est, confidence_level=self.confidence_level,
            conformity_score="lac", prefit=True,
        )
        self._mapie.conformalize(idx, y_cal)
        scores = np.asarray(self._mapie.conformity_scores).reshape(-1)
        n = len(scores)
        alpha = 1.0 - self.confidence_level
        q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        self._threshold = float(np.quantile(scores, q_level, method="higher"))
        self._n_classes = proba_cal.shape[1]

    def _calibrate_numpy(self, proba_cal: np.ndarray, y_cal: np.ndarray) -> None:
        # LAC score s_i = 1 - p_i[true label]. The (1-alpha) quantile with the
        # standard finite-sample correction is the set-inclusion threshold.
        n = len(y_cal)
        scores = 1.0 - proba_cal[np.arange(n), y_cal]
        alpha = 1.0 - self.confidence_level
        q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        self._threshold = float(np.quantile(scores, q_level, method="higher"))
        self._n_classes = proba_cal.shape[1]

    # -- prediction -----------------------------------------------------
    def predict_set(self, proba: np.ndarray) -> np.ndarray:
        """Return a boolean (N, K) mask: True where a class is in the set.

        Both backends apply the LAC set rule with the calibrated threshold:
        a class is in the set iff ``1 - p <= threshold`` (i.e. ``p >= 1 - thr``).
        """
        proba = np.asarray(proba, dtype=np.float64)
        assert self._threshold is not None, "calibrate() first"
        return proba >= (1.0 - self._threshold)

    # -- reporting ------------------------------------------------------
    @staticmethod
    def empirical_coverage(sets: np.ndarray, y_true: np.ndarray) -> float:
        """Fraction of samples whose true label is in the predicted set."""
        y_true = np.asarray(y_true).astype(int)
        hit = sets[np.arange(len(y_true)), y_true]
        return float(hit.mean())

    @staticmethod
    def mean_set_size(sets: np.ndarray) -> float:
        """Average number of classes per prediction set (efficiency)."""
        return float(sets.sum(axis=1).mean())
