"""
SAMYUKTA — Meteorological Forecast Calibration Module

Provides statistical and probabilistic calibration for multi-model weather forecasts:
1. Probability Calibration: Platt scaling (logistic) & Isotonic regression for hazard alerts
2. Empirical Quantile Mapping (EQM): Non-parametric distribution matching for rainfall & temperature
3. Spread-Error Calibration (Variance Scaling): Calibrates ensemble spread to match forecast RMSE
"""
from __future__ import annotations

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Any
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Evaluation metrics for a calibrated variable or probability."""
    variable: str
    method: str
    brier_score_before: float | None = None
    brier_score_after: float | None = None
    rmse_before: float | None = None
    rmse_after: float | None = None
    spread_error_ratio: float = 1.0

    def to_dict(self) -> dict:
        return {
            "variable": self.variable,
            "method": self.method,
            "brier_score_before": round(self.brier_score_before, 4) if self.brier_score_before is not None else None,
            "brier_score_after": round(self.brier_score_after, 4) if self.brier_score_after is not None else None,
            "rmse_before": round(self.rmse_before, 4) if self.rmse_before is not None else None,
            "rmse_after": round(self.rmse_after, 4) if self.rmse_after is not None else None,
            "spread_error_ratio": round(self.spread_error_ratio, 4),
        }


class ProbabilityCalibrator:
    """
    Calibrates binary event probabilities (e.g., Heavy Rain > 64.5 mm, Heat Wave > 40°C).
    Supports Platt scaling (logistic regression) and Isotonic Regression.
    """

    def __init__(self, method: str = "platt"):
        self.method = method
        self.calibrator: Any = None
        self.is_fitted = False

    def fit(self, raw_probs: np.ndarray, event_occurred: np.ndarray) -> "ProbabilityCalibrator":
        """
        Fit probability calibrator using historical predicted probabilities and outcomes.
        """
        raw_probs = np.clip(np.asarray(raw_probs, dtype=np.float64).ravel(), 1e-6, 1.0 - 1e-6)
        labels = np.asarray(event_occurred, dtype=np.int32).ravel()

        if len(np.unique(labels)) < 2:
            # Degenerate case (only 0s or only 1s observed)
            self.is_fitted = False
            return self

        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(raw_probs, labels)
        else:  # Platt scaling
            self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            # Log-odds input for Platt scaling
            log_odds = np.log(raw_probs / (1.0 - raw_probs)).reshape(-1, 1)
            self.calibrator.fit(log_odds, labels)

        self.is_fitted = True
        return self

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply calibration to raw probability values."""
        arr = np.clip(np.asarray(raw_probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        if not self.is_fitted or self.calibrator is None:
            return arr

        shape = arr.shape
        flat = arr.ravel()

        if self.method == "isotonic":
            calibrated = self.calibrator.predict(flat)
        else:
            log_odds = np.log(flat / (1.0 - flat)).reshape(-1, 1)
            calibrated = self.calibrator.predict_proba(log_odds)[:, 1]

        return np.clip(calibrated.reshape(shape), 0.0, 1.0)


class QuantileMappingCalibrator:
    """
    Empirical Quantile Mapping (EQM).
    Calibrates continuous forecast fields against historical climatological observations.
    """

    def __init__(self, n_quantiles: int = 100):
        self.n_quantiles = n_quantiles
        self.forecast_quantiles: np.ndarray | None = None
        self.obs_quantiles: np.ndarray | None = None
        self.quantiles = np.linspace(0.0, 1.0, n_quantiles)
        self.is_fitted = False

    def fit(self, historical_forecasts: np.ndarray, historical_obs: np.ndarray) -> "QuantileMappingCalibrator":
        """Calculate transfer function between forecast and observed empirical CDFs."""
        fc_clean = historical_forecasts[~np.isnan(historical_forecasts)].ravel()
        obs_clean = historical_obs[~np.isnan(historical_obs)].ravel()

        if len(fc_clean) < 10 or len(obs_clean) < 10:
            self.is_fitted = False
            return self

        self.forecast_quantiles = np.quantile(fc_clean, self.quantiles)
        self.obs_quantiles = np.quantile(obs_clean, self.quantiles)
        self.is_fitted = True
        return self

    def calibrate(self, forecast: np.ndarray) -> np.ndarray:
        """Map forecast values through the empirical quantile mapping transfer function."""
        if not self.is_fitted or self.forecast_quantiles is None or self.obs_quantiles is None:
            return forecast

        arr = np.asarray(forecast, dtype=np.float64)
        # Interpolate percentile rank of forecast, then interpolate matching observed quantile
        ranks = np.interp(arr, self.forecast_quantiles, self.quantiles)
        calibrated = np.interp(ranks, self.quantiles, self.obs_quantiles)
        return calibrated


class SpreadCalibrator:
    """
    Calibrates ensemble spread to match RMSE (Spread-Skill Relationship).
    Variance scaling factor s = mean(RMSE) / mean(Spread).
    """

    def __init__(self):
        self.scale_factor: float = 1.0
        self.is_fitted = False

    def fit(self, ensemble_spreads: np.ndarray, forecast_errors: np.ndarray) -> "SpreadCalibrator":
        """Compute the variance inflation/deflation factor."""
        spreads = np.asarray(ensemble_spreads).ravel()
        errors = np.abs(np.asarray(forecast_errors).ravel())

        mask = ~np.isnan(spreads) & ~np.isnan(errors) & (spreads > 1e-4)
        if np.sum(mask) < 5:
            self.scale_factor = 1.0
            self.is_fitted = False
            return self

        mean_spread = float(np.mean(spreads[mask]))
        mean_error = float(np.mean(errors[mask]))

        self.scale_factor = float(np.clip(mean_error / max(mean_spread, 1e-6), 0.5, 3.0))
        self.is_fitted = True
        return self

    def calibrate_spread(self, spread: float | np.ndarray) -> float | np.ndarray:
        """Scale spread by calibrated variance factor."""
        return spread * self.scale_factor


class CalibrationEngine:
    """
    Master Calibration Engine for SAMYUKTA.
    Coordinates quantile mapping, probability calibration, and spread scaling across variables.
    """

    def __init__(self):
        self.quantile_calibrators: dict[str, QuantileMappingCalibrator] = {}
        self.prob_calibrators: dict[str, ProbabilityCalibrator] = {}
        self.spread_calibrators: dict[str, SpreadCalibrator] = {}
        self.metrics_history: list[CalibrationMetrics] = []

    def fit_variable(
        self,
        variable: str,
        historical_fc: np.ndarray,
        historical_obs: np.ndarray,
        threshold: float | None = None,
    ) -> CalibrationMetrics:
        """Fit all calibration models for a given weather variable."""
        # 1. Quantile mapping
        qm = QuantileMappingCalibrator()
        qm.fit(historical_fc, historical_obs)
        self.quantile_calibrators[variable] = qm

        # Compute RMSE before and after QM
        raw_rmse = float(np.sqrt(np.nanmean((historical_fc - historical_obs) ** 2)))
        cal_fc = qm.calibrate(historical_fc)
        cal_rmse = float(np.sqrt(np.nanmean((cal_fc - historical_obs) ** 2)))

        # 2. Probability calibration if threshold is given
        brier_before, brier_after = None, None
        if threshold is not None:
            raw_event_prob = np.clip(historical_fc / max(threshold * 1.5, 1.0), 0.0, 1.0)
            actual_events = (historical_obs >= threshold).astype(int)
            brier_before = float(np.mean((raw_event_prob - actual_events) ** 2))

            pc = ProbabilityCalibrator(method="platt")
            pc.fit(raw_event_prob, actual_events)
            self.prob_calibrators[variable] = pc

            cal_prob = pc.calibrate(raw_event_prob)
            brier_after = float(np.mean((cal_prob - actual_events) ** 2))

        # 3. Spread calibrator
        errors = historical_fc - historical_obs
        spread_proxy = np.abs(errors) * 0.8
        sc = SpreadCalibrator()
        sc.fit(spread_proxy, errors)
        self.spread_calibrators[variable] = sc

        metrics = CalibrationMetrics(
            variable=variable,
            method="eqm_platt",
            brier_score_before=brier_before,
            brier_score_after=brier_after,
            rmse_before=raw_rmse,
            rmse_after=cal_rmse,
            spread_error_ratio=sc.scale_factor,
        )
        self.metrics_history.append(metrics)
        return metrics

    def calibrate_forecast(self, forecast: np.ndarray, variable: str) -> np.ndarray:
        """Apply quantile mapping calibration to forecast field."""
        if variable in self.quantile_calibrators:
            return self.quantile_calibrators[variable].calibrate(forecast)
        return forecast

    def calibrate_probability(self, raw_prob: float | np.ndarray, variable: str) -> float | np.ndarray:
        """Calibrate hazard/event probability."""
        if variable in self.prob_calibrators:
            return self.prob_calibrators[variable].calibrate(raw_prob)
        return raw_prob

    def calibrate_spread(self, spread: float | np.ndarray, variable: str) -> float | np.ndarray:
        """Calibrate ensemble spread."""
        if variable in self.spread_calibrators:
            return self.spread_calibrators[variable].calibrate_spread(spread)
        return spread
