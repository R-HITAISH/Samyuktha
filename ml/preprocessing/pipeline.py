"""
SAMYUKTA — Data Preprocessing & Quality Engine

Handles spatial regridding, temporal alignment, quality control, and
bias correction before forecast blending.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass, field
from typing import Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Quality assessment for a single model/source."""
    model_id: str
    coverage: float           # fraction of non-NaN values
    missing_fraction: float   # fraction of missing data
    has_nans: bool
    has_duplicates: bool
    value_range_valid: bool
    stale: bool
    quality_score: float      # 0-1 composite
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "coverage": round(self.coverage, 4),
            "missing_fraction": round(self.missing_fraction, 4),
            "has_nans": self.has_nans,
            "has_duplicates": self.has_duplicates,
            "value_range_valid": self.value_range_valid,
            "stale": self.stale,
            "quality_score": round(self.quality_score, 4),
            "issues": self.issues,
        }


class DataQualityEngine:
    """
    Validates and scores data quality for each model source.
    Quality score becomes an input to the blending engine.
    """

    VALID_RANGES = {
        "rainfall": (0, 1000),
        "temperature": (-40, 60),
        "wind_speed": (0, 120),
    }

    def __init__(
        self,
        min_coverage: float = 0.7,
        max_missing: float = 0.3,
        stale_hours: float = 12,
        outlier_sigma: float = 5.0,
    ):
        self.min_coverage = min_coverage
        self.max_missing = max_missing
        self.stale_hours = stale_hours
        self.outlier_sigma = outlier_sigma

    def assess(self, ds: xr.Dataset, model_id: str, reference_time: pd.Timestamp = None) -> QualityReport:
        """Run full quality assessment on a dataset."""
        issues = []

        # Coverage
        total = sum(ds[v].size for v in ds.data_vars)
        nan_count = sum(int(np.isnan(ds[v].values).sum()) for v in ds.data_vars)
        coverage = 1.0 - (nan_count / max(total, 1))
        missing_frac = nan_count / max(total, 1)
        has_nans = nan_count > 0

        if has_nans:
            issues.append(f"Contains {nan_count} NaN values")
        if coverage < self.min_coverage:
            issues.append(f"Coverage {coverage:.2%} below minimum {self.min_coverage:.2%}")

        # Duplicate timestamps
        has_dupes = False
        if "time" in ds.coords:
            times = pd.DatetimeIndex(ds.time.values)
            has_dupes = times.duplicated().any()
            if has_dupes:
                issues.append("Duplicate timestamps detected")

        # Value range check
        range_valid = True
        for var in ds.data_vars:
            if var in self.VALID_RANGES:
                vmin, vmax = self.VALID_RANGES[var]
                vals = ds[var].values
                valid_vals = vals[~np.isnan(vals)]
                if len(valid_vals) > 0:
                    if valid_vals.min() < vmin or valid_vals.max() > vmax:
                        range_valid = False
                        issues.append(
                            f"{var}: values [{valid_vals.min():.1f}, {valid_vals.max():.1f}] "
                            f"outside valid range [{vmin}, {vmax}]"
                        )

        # Staleness check
        stale = False
        if reference_time and "time" in ds.coords:
            latest = pd.Timestamp(ds.time.values[-1])
            hours_old = (reference_time - latest).total_seconds() / 3600
            if hours_old > self.stale_hours:
                stale = True
                issues.append(f"Data is {hours_old:.1f} hours old (threshold: {self.stale_hours}h)")

        # Outlier check
        for var in ds.data_vars:
            vals = ds[var].values[~np.isnan(ds[var].values)]
            if len(vals) > 10:
                mean, std = vals.mean(), vals.std()
                if std > 0:
                    n_outliers = int(np.sum(np.abs(vals - mean) > self.outlier_sigma * std))
                    if n_outliers > 0:
                        issues.append(f"{var}: {n_outliers} outliers (>{self.outlier_sigma}σ)")

        # Composite quality score
        score = 1.0
        score *= coverage
        if not range_valid:
            score *= 0.8
        if stale:
            score *= 0.7
        if has_dupes:
            score *= 0.9
        score = max(0.0, min(1.0, score))

        return QualityReport(
            model_id=model_id,
            coverage=coverage,
            missing_fraction=missing_frac,
            has_nans=has_nans,
            has_duplicates=has_dupes,
            value_range_valid=range_valid,
            stale=stale,
            quality_score=score,
            issues=issues,
        )


class SpatialRegridder:
    """
    Regrids forecast data to a common target grid.
    Supports nearest-neighbor and bilinear interpolation.
    """

    def __init__(self, target_lats: np.ndarray, target_lons: np.ndarray, method: str = "linear"):
        self.target_lats = target_lats
        self.target_lons = target_lons
        self.method = method  # 'linear', 'nearest'

    def regrid(self, ds: xr.Dataset) -> xr.Dataset:
        """Regrid dataset to target grid."""
        source_lats = ds.lat.values
        source_lons = ds.lon.values

        # Check if already on target grid
        if (np.allclose(source_lats, self.target_lats, atol=0.01) and
            np.allclose(source_lons, self.target_lons, atol=0.01)):
            return ds

        data_vars = {}
        for var in ds.data_vars:
            arr = ds[var].values
            if arr.ndim == 3:  # (time, lat, lon)
                regridded = np.zeros(
                    (arr.shape[0], len(self.target_lats), len(self.target_lons)),
                    dtype=np.float32,
                )
                for t in range(arr.shape[0]):
                    interp = RegularGridInterpolator(
                        (source_lats, source_lons),
                        arr[t],
                        method=self.method,
                        bounds_error=False,
                        fill_value=np.nan,
                    )
                    target_grid = np.meshgrid(self.target_lats, self.target_lons, indexing="ij")
                    points = np.stack([target_grid[0].ravel(), target_grid[1].ravel()], axis=-1)
                    regridded[t] = interp(points).reshape(len(self.target_lats), len(self.target_lons))
                data_vars[var] = (["time", "lat", "lon"], regridded)
            elif arr.ndim == 2:  # (lat, lon)
                interp = RegularGridInterpolator(
                    (source_lats, source_lons),
                    arr,
                    method=self.method,
                    bounds_error=False,
                    fill_value=np.nan,
                )
                target_grid = np.meshgrid(self.target_lats, self.target_lons, indexing="ij")
                points = np.stack([target_grid[0].ravel(), target_grid[1].ravel()], axis=-1)
                regridded = interp(points).reshape(len(self.target_lats), len(self.target_lons))
                data_vars[var] = (["lat", "lon"], regridded.astype(np.float32))

        coords = dict(ds.coords)
        coords["lat"] = self.target_lats
        coords["lon"] = self.target_lons
        return xr.Dataset(data_vars, coords=coords, attrs=ds.attrs)


class TemporalAligner:
    """
    Ensures all forecasts use consistent temporal referencing.
    Stores init_time, valid_time, and lead_time_hours explicitly.
    """

    @staticmethod
    def align(ds: xr.Dataset, init_time: pd.Timestamp, lead_time_hours: int) -> xr.Dataset:
        """Add explicit temporal metadata."""
        ds = ds.copy()
        ds.attrs["init_time"] = init_time.isoformat()
        ds.attrs["lead_time_hours"] = lead_time_hours
        # Calculate valid times
        if "time" in ds.coords:
            valid_times = pd.DatetimeIndex(ds.time.values)
            ds.attrs["valid_time_start"] = valid_times[0].isoformat()
            ds.attrs["valid_time_end"] = valid_times[-1].isoformat()
        return ds


class BiasCorrector:
    """
    Model-specific bias correction using linear correction or quantile mapping.

    Learns correction parameters from training data only.
    """

    def __init__(self):
        self.corrections: dict[str, dict[str, dict]] = {}  # model -> var -> params

    def fit(
        self,
        model_id: str,
        forecast: xr.Dataset,
        observation: xr.Dataset,
        variables: list[str] = None,
    ) -> None:
        """Learn bias correction parameters from paired forecast-observation data."""
        if variables is None:
            variables = [v for v in forecast.data_vars if v in observation.data_vars]

        self.corrections.setdefault(model_id, {})

        for var in variables:
            fc = forecast[var].values.ravel()
            obs = observation[var].values.ravel()

            # Remove NaNs
            mask = ~(np.isnan(fc) | np.isnan(obs))
            fc_clean = fc[mask]
            obs_clean = obs[mask]

            if len(fc_clean) < 10:
                continue

            # Linear bias correction: obs = a * fc + b
            bias = np.mean(fc_clean) - np.mean(obs_clean)
            scale = np.std(obs_clean) / max(np.std(fc_clean), 1e-8)

            # Quantile mapping parameters
            n_quantiles = 100
            fc_quantiles = np.percentile(fc_clean, np.linspace(0, 100, n_quantiles))
            obs_quantiles = np.percentile(obs_clean, np.linspace(0, 100, n_quantiles))

            self.corrections[model_id][var] = {
                "bias": float(bias),
                "scale": float(scale),
                "mean_fc": float(np.mean(fc_clean)),
                "mean_obs": float(np.mean(obs_clean)),
                "fc_quantiles": fc_quantiles.tolist(),
                "obs_quantiles": obs_quantiles.tolist(),
                "method": "linear",  # default
                "n_samples": len(fc_clean),
            }

        logger.info(f"Fit bias correction for {model_id}: {list(self.corrections[model_id].keys())}")

    def correct(
        self,
        model_id: str,
        forecast: xr.Dataset,
        method: str = "linear",
    ) -> xr.Dataset:
        """Apply bias correction to forecast."""
        if model_id not in self.corrections:
            logger.warning(f"No correction params for {model_id}, returning raw forecast")
            return forecast

        ds = forecast.copy(deep=True)
        for var in ds.data_vars:
            if var not in self.corrections[model_id]:
                continue

            params = self.corrections[model_id][var]
            arr = ds[var].values

            if method == "linear":
                corrected = (arr - params["mean_fc"]) * params["scale"] + params["mean_obs"]
            elif method == "quantile":
                fc_q = np.array(params["fc_quantiles"])
                obs_q = np.array(params["obs_quantiles"])
                corrected = np.interp(arr.ravel(), fc_q, obs_q).reshape(arr.shape)
            else:
                corrected = arr - params["bias"]

            # Apply physical constraints
            if var == "rainfall":
                corrected = np.clip(corrected, 0, None)
            elif var == "wind_speed":
                corrected = np.clip(corrected, 0, None)

            ds[var].values = corrected.astype(np.float32)

        ds.attrs["bias_corrected"] = True
        ds.attrs["bias_correction_method"] = method
        return ds

    def get_corrections(self) -> dict:
        """Return correction parameters for audit."""
        return self.corrections


class PhysicsValidator:
    """Post-processing validation to enforce physical constraints."""

    CONSTRAINTS = {
        "rainfall": {"min": 0, "max": 1000},
        "temperature": {"min": -40, "max": 60},
        "wind_speed": {"min": 0, "max": 120},
        "probability": {"min": 0, "max": 1},
        "weight": {"min": 0, "max": 1},
    }

    @classmethod
    def validate(cls, ds: xr.Dataset) -> xr.Dataset:
        """Clip values to physically valid ranges."""
        ds = ds.copy(deep=True)
        for var in ds.data_vars:
            if var in cls.CONSTRAINTS:
                c = cls.CONSTRAINTS[var]
                ds[var].values = np.clip(ds[var].values, c["min"], c["max"])
        return ds

    @classmethod
    def validate_weights(cls, weights: np.ndarray) -> np.ndarray:
        """Ensure weights are non-negative and sum to 1."""
        weights = np.clip(weights, 0, None)
        weight_sum = weights.sum(axis=0, keepdims=True)
        weight_sum = np.where(weight_sum == 0, 1, weight_sum)
        return weights / weight_sum
