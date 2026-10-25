"""
SAMYUKTA — Uncertainty Quantification Module

Provides calibrated prediction intervals using:
- Ensemble spread (primary method)
- Conformal prediction
- Quantile regression (adapter-ready)

Ensures non-negative rainfall and bounded probabilities.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyEstimate:
    """Uncertainty quantification result."""
    deterministic: float
    lower_bound: float
    upper_bound: float
    confidence_level: float  # e.g., 0.95 for 95% interval
    uncertainty_score: float  # 0=certain, 1=very uncertain
    method: str
    spread: float
    variable: str = ""

    def to_dict(self) -> dict:
        return {
            "deterministic": round(self.deterministic, 3),
            "lower_bound": round(self.lower_bound, 3),
            "upper_bound": round(self.upper_bound, 3),
            "confidence_level": self.confidence_level,
            "uncertainty_score": round(self.uncertainty_score, 4),
            "method": self.method,
            "spread": round(self.spread, 3),
            "variable": self.variable,
        }


class UncertaintyEngine:
    """
    Quantifies forecast uncertainty from multiple sources.

    Methods:
    1. Ensemble spread — standard deviation across models
    2. Weighted spread — skill-weighted model disagreement
    3. Conformal prediction — calibrated intervals from historical residuals
    """

    def __init__(self, confidence_level: float = 0.90):
        self.confidence_level = confidence_level
        self.calibration_residuals: dict[str, np.ndarray] = {}

    def from_ensemble_spread(
        self,
        model_forecasts: np.ndarray,
        weights: np.ndarray = None,
        variable: str = "rainfall",
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty from multi-model spread.

        For rainfall: uses asymmetric intervals (no negative values).
        """
        if weights is not None:
            weighted_mean = np.sum(weights * model_forecasts)
            weighted_var = np.sum(weights * (model_forecasts - weighted_mean) ** 2)
            spread = np.sqrt(weighted_var)
        else:
            weighted_mean = np.mean(model_forecasts)
            spread = np.std(model_forecasts)

        # Z-score for confidence level
        from scipy.stats import norm
        z = norm.ppf(0.5 + self.confidence_level / 2)

        lower = weighted_mean - z * spread
        upper = weighted_mean + z * spread

        # Physics constraints
        if variable in ("rainfall", "wind_speed"):
            lower = max(lower, 0)

        # Uncertainty score (0-1): normalized spread relative to mean
        unc_score = float(np.clip(spread / max(abs(weighted_mean), 1), 0, 1))

        return UncertaintyEstimate(
            deterministic=float(weighted_mean),
            lower_bound=float(lower),
            upper_bound=float(upper),
            confidence_level=self.confidence_level,
            uncertainty_score=unc_score,
            method="ensemble_spread",
            spread=float(spread),
            variable=variable,
        )

    def calibrate_conformal(
        self,
        residuals: np.ndarray,
        variable: str,
    ) -> None:
        """
        Store historical residuals for conformal prediction.
        Residuals = forecast - observation.
        """
        self.calibration_residuals[variable] = np.sort(np.abs(residuals))
        logger.info(f"Conformal calibration stored for {variable}: {len(residuals)} residuals")

    def from_conformal(
        self,
        forecast: float,
        variable: str = "rainfall",
    ) -> UncertaintyEstimate:
        """
        Conformal prediction interval using stored calibration residuals.
        """
        if variable not in self.calibration_residuals:
            # Fallback to simple estimate
            spread = abs(forecast) * 0.2
            return UncertaintyEstimate(
                deterministic=forecast,
                lower_bound=max(forecast - spread, 0) if variable in ("rainfall", "wind_speed") else forecast - spread,
                upper_bound=forecast + spread,
                confidence_level=self.confidence_level,
                uncertainty_score=0.3,
                method="conformal_fallback",
                spread=spread,
                variable=variable,
            )

        residuals = self.calibration_residuals[variable]
        n = len(residuals)
        idx = int(np.ceil((n + 1) * self.confidence_level)) - 1
        idx = min(idx, n - 1)
        q = residuals[idx]

        lower = forecast - q
        upper = forecast + q
        if variable in ("rainfall", "wind_speed"):
            lower = max(lower, 0)

        return UncertaintyEstimate(
            deterministic=forecast,
            lower_bound=float(lower),
            upper_bound=float(upper),
            confidence_level=self.confidence_level,
            uncertainty_score=float(np.clip(q / max(abs(forecast), 1), 0, 1)),
            method="conformal_prediction",
            spread=float(q),
            variable=variable,
        )

    def spatial_uncertainty(
        self,
        model_grids: dict[str, np.ndarray],
        weights: dict[str, np.ndarray] = None,
        variable: str = "rainfall",
    ) -> dict[str, np.ndarray]:
        """
        Compute spatial uncertainty maps.
        Returns deterministic, lower, upper, and uncertainty_score grids.
        """
        stack = np.stack(list(model_grids.values()))
        mean = np.mean(stack, axis=0)
        std = np.std(stack, axis=0)

        from scipy.stats import norm
        z = norm.ppf(0.5 + self.confidence_level / 2)

        lower = mean - z * std
        upper = mean + z * std

        if variable in ("rainfall", "wind_speed"):
            lower = np.clip(lower, 0, None)

        unc_score = std / np.maximum(np.abs(mean), 1)
        unc_score = np.clip(unc_score, 0, 1)

        return {
            "deterministic": mean,
            "lower_bound": lower,
            "upper_bound": upper,
            "uncertainty_score": unc_score,
            "spread": std,
        }
