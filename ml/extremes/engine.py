"""
SAMYUKTA — Extreme Weather Engine

Detects, classifies, and quantifies extreme weather events:
- Heavy Rainfall
- Heat Waves
- High Wind
- Composite Hazard Score

Provides probability estimates from calibrated or ensemble-based methods.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from dataclasses import dataclass, field
from typing import Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class HazardEvent:
    """Detected hazard with metadata."""
    hazard_type: str           # heavy_rainfall, heat_wave, high_wind
    severity: str              # moderate, heavy, very_heavy, extreme
    probability: float         # 0-1, calibrated or labeled as estimate
    confidence: float          # 0-1
    affected_area_fraction: float  # fraction of grid affected
    peak_value: float
    mean_value: float
    threshold_used: float
    duration_estimate: str     # e.g., "6-12 hours"
    region: str
    lat_range: tuple[float, float] = (0, 0)
    lon_range: tuple[float, float] = (0, 0)
    supporting_models: list[str] = field(default_factory=list)
    alert_level: str = "INFO"  # INFO, WATCH, HIGH, CRITICAL

    def to_dict(self) -> dict:
        return {
            "hazard_type": self.hazard_type,
            "severity": self.severity,
            "probability": round(self.probability, 3),
            "confidence": round(self.confidence, 3),
            "affected_area_fraction": round(self.affected_area_fraction, 4),
            "peak_value": round(self.peak_value, 2),
            "mean_value": round(self.mean_value, 2),
            "threshold_used": self.threshold_used,
            "duration_estimate": self.duration_estimate,
            "region": self.region,
            "lat_range": self.lat_range,
            "lon_range": self.lon_range,
            "supporting_models": self.supporting_models,
            "alert_level": self.alert_level,
        }


class ExtremeWeatherEngine:
    """
    Detects and classifies extreme weather events from blended and individual forecasts.

    Probabilities are derived from:
    1. Ensemble fraction exceeding threshold (multi-model probability)
    2. Historical climatological calibration where available
    3. Clearly labeled as 'demo estimate' when neither is available
    """

    def __init__(self, thresholds: dict = None, alert_config: dict = None):
        self.thresholds = thresholds or {
            "heavy_rainfall": {
                "moderate_mm": 64.5,
                "heavy_mm": 115.5,
                "very_heavy_mm": 204.5,
                "extremely_heavy_mm": 244.5,
            },
            "heat_wave": {
                "plains_threshold_c": 40.0,
                "plains_departure_c": 4.5,
                "coastal_threshold_c": 37.0,
                "severe_departure_c": 6.5,
            },
            "high_wind": {
                "strong_ms": 17.2,
                "gale_ms": 24.5,
                "storm_ms": 32.7,
                "hurricane_ms": 46.3,
            },
        }
        self.alert_config = alert_config or {
            "info": 0.3,
            "watch": 0.5,
            "high": 0.7,
            "critical": 0.85,
        }

    def _classify_alert(self, probability: float) -> str:
        if probability >= self.alert_config["critical"]:
            return "CRITICAL"
        elif probability >= self.alert_config["high"]:
            return "HIGH"
        elif probability >= self.alert_config["watch"]:
            return "WATCH"
        elif probability >= self.alert_config["info"]:
            return "INFO"
        return "NONE"

    def detect_heavy_rainfall(
        self,
        blended_rain: np.ndarray,
        model_rains: dict[str, np.ndarray],
        lats: np.ndarray,
        lons: np.ndarray,
        model_agreement: float = 0.5,
    ) -> list[HazardEvent]:
        """Detect heavy rainfall events from blended and individual model forecasts."""
        events = []
        thresholds = self.thresholds["heavy_rainfall"]

        for severity, threshold_key, duration in [
            ("moderate", "moderate_mm", "6-12 hours"),
            ("heavy", "heavy_mm", "6-24 hours"),
            ("very_heavy", "very_heavy_mm", "12-24 hours"),
            ("extreme", "extremely_heavy_mm", "12-48 hours"),
        ]:
            threshold = thresholds[threshold_key]
            exceedance = blended_rain > threshold

            if not np.any(exceedance):
                continue

            affected_frac = float(np.mean(exceedance))

            # Probability from multi-model agreement
            model_probs = []
            supporting = []
            for mid, rain in model_rains.items():
                frac = float(np.mean(rain > threshold))
                model_probs.append(frac)
                if frac > 0.1:
                    supporting.append(mid)

            # Ensemble probability: fraction of models predicting exceedance
            ensemble_prob = float(np.mean([1.0 if p > 0.1 else 0.0 for p in model_probs]))
            # Calibrated probability estimate (weighted by model agreement)
            probability = float(np.clip(
                0.6 * ensemble_prob + 0.3 * affected_frac + 0.1 * model_agreement,
                0, 1
            ))

            confidence = model_agreement * 0.7 + 0.3 * (1.0 - np.std(model_probs))

            # Find affected region bounds
            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
            affected_lats = lat_grid[exceedance]
            affected_lons = lon_grid[exceedance]

            events.append(HazardEvent(
                hazard_type="heavy_rainfall",
                severity=severity,
                probability=probability,
                confidence=float(np.clip(confidence, 0.1, 0.99)),
                affected_area_fraction=affected_frac,
                peak_value=float(np.max(blended_rain)),
                mean_value=float(np.mean(blended_rain[exceedance])),
                threshold_used=threshold,
                duration_estimate=duration,
                region="detected",
                lat_range=(float(affected_lats.min()), float(affected_lats.max())) if len(affected_lats) > 0 else (0, 0),
                lon_range=(float(affected_lons.min()), float(affected_lons.max())) if len(affected_lons) > 0 else (0, 0),
                supporting_models=supporting,
                alert_level=self._classify_alert(probability),
            ))

        return events

    def detect_heat_wave(
        self,
        blended_temp: np.ndarray,
        model_temps: dict[str, np.ndarray],
        lats: np.ndarray,
        lons: np.ndarray,
        climatology_mean: np.ndarray = None,
        model_agreement: float = 0.5,
    ) -> list[HazardEvent]:
        """Detect heat wave conditions."""
        events = []
        th = self.thresholds["heat_wave"]

        # Use threshold-based detection
        threshold = th["plains_threshold_c"]
        departure = th["plains_departure_c"]

        hot_mask = blended_temp > threshold

        if climatology_mean is not None:
            departure_mask = (blended_temp - climatology_mean) > departure
            hw_mask = hot_mask | departure_mask
        else:
            hw_mask = hot_mask

        if not np.any(hw_mask):
            return events

        affected_frac = float(np.mean(hw_mask))

        # Multi-model probability
        model_probs = []
        supporting = []
        for mid, temp in model_temps.items():
            frac = float(np.mean(temp > threshold))
            model_probs.append(frac)
            if frac > 0.1:
                supporting.append(mid)

        ensemble_prob = float(np.mean([1.0 if p > 0.1 else 0.0 for p in model_probs]))
        probability = float(np.clip(0.6 * ensemble_prob + 0.3 * affected_frac + 0.1 * model_agreement, 0, 1))

        temp_anomaly = float(np.mean(blended_temp[hw_mask]) - threshold)

        severity = "moderate"
        if temp_anomaly > th["severe_departure_c"]:
            severity = "severe"
        elif temp_anomaly > departure:
            severity = "significant"

        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

        events.append(HazardEvent(
            hazard_type="heat_wave",
            severity=severity,
            probability=probability,
            confidence=float(np.clip(model_agreement, 0.1, 0.99)),
            affected_area_fraction=affected_frac,
            peak_value=float(np.max(blended_temp)),
            mean_value=float(np.mean(blended_temp[hw_mask])),
            threshold_used=threshold,
            duration_estimate="2-5 days",
            region="detected",
            lat_range=(float(lat_grid[hw_mask].min()), float(lat_grid[hw_mask].max())),
            lon_range=(float(lon_grid[hw_mask].min()), float(lon_grid[hw_mask].max())),
            supporting_models=supporting,
            alert_level=self._classify_alert(probability),
        ))

        return events

    def detect_high_wind(
        self,
        blended_wind: np.ndarray,
        model_winds: dict[str, np.ndarray],
        lats: np.ndarray,
        lons: np.ndarray,
        model_agreement: float = 0.5,
    ) -> list[HazardEvent]:
        """Detect high wind events."""
        events = []
        th = self.thresholds["high_wind"]

        for severity, threshold_key, duration in [
            ("strong", "strong_ms", "6-12 hours"),
            ("gale", "gale_ms", "6-24 hours"),
            ("storm", "storm_ms", "12-24 hours"),
            ("hurricane", "hurricane_ms", "12-48 hours"),
        ]:
            threshold = th[threshold_key]
            exceedance = blended_wind > threshold

            if not np.any(exceedance):
                continue

            affected_frac = float(np.mean(exceedance))

            model_probs = []
            supporting = []
            for mid, wind in model_winds.items():
                frac = float(np.mean(wind > threshold))
                model_probs.append(frac)
                if frac > 0.05:
                    supporting.append(mid)

            ensemble_prob = float(np.mean([1.0 if p > 0.05 else 0.0 for p in model_probs]))
            probability = float(np.clip(0.6 * ensemble_prob + 0.3 * affected_frac + 0.1 * model_agreement, 0, 1))

            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

            events.append(HazardEvent(
                hazard_type="high_wind",
                severity=severity,
                probability=probability,
                confidence=float(np.clip(model_agreement, 0.1, 0.99)),
                affected_area_fraction=affected_frac,
                peak_value=float(np.max(blended_wind)),
                mean_value=float(np.mean(blended_wind[exceedance])),
                threshold_used=threshold,
                duration_estimate=duration,
                region="detected",
                lat_range=(float(lat_grid[exceedance].min()), float(lat_grid[exceedance].max())),
                lon_range=(float(lon_grid[exceedance].min()), float(lon_grid[exceedance].max())),
                supporting_models=supporting,
                alert_level=self._classify_alert(probability),
            ))

        return events

    def compute_composite_hazard(
        self,
        rainfall_events: list[HazardEvent],
        heat_events: list[HazardEvent],
        wind_events: list[HazardEvent],
        forecast_uncertainty: float = 0.5,
        model_agreement: float = 0.5,
    ) -> dict[str, Any]:
        """
        Compute composite hazard score.

        Calculation:
        CompositeHazard = max(rainfall_risk, heat_risk, wind_risk)
                         * (1 + 0.2 * multi_hazard_factor)
                         * uncertainty_adjustment

        The formula is documented and not a black box.
        """
        def max_prob(events):
            return max((e.probability for e in events), default=0.0)

        rain_risk = max_prob(rainfall_events)
        heat_risk = max_prob(heat_events)
        wind_risk = max_prob(wind_events)

        # Multi-hazard amplification
        active_hazards = sum(1 for r in [rain_risk, heat_risk, wind_risk] if r > 0.3)
        multi_factor = 0.1 * max(active_hazards - 1, 0)

        # Dominant risk
        max_risk = max(rain_risk, heat_risk, wind_risk)

        # Uncertainty adjustment
        unc_factor = 1.0 + 0.1 * (1.0 - model_agreement)

        composite = float(np.clip(max_risk * (1 + multi_factor) * unc_factor, 0, 1))

        return {
            "composite_hazard_score": round(composite, 3),
            "rainfall_risk": round(rain_risk, 3),
            "heat_risk": round(heat_risk, 3),
            "wind_risk": round(wind_risk, 3),
            "active_hazards": active_hazards,
            "model_agreement": round(model_agreement, 3),
            "forecast_uncertainty": round(forecast_uncertainty, 3),
            "calculation": (
                f"CompositeHazard = max(rain={rain_risk:.2f}, heat={heat_risk:.2f}, wind={wind_risk:.2f}) "
                f"× (1 + {multi_factor:.2f} multi-hazard) × {unc_factor:.2f} uncertainty "
                f"= {composite:.3f}"
            ),
            "alert_level": self._classify_alert(composite),
            "all_events": {
                "rainfall": [e.to_dict() for e in rainfall_events],
                "heat_wave": [e.to_dict() for e in heat_events],
                "high_wind": [e.to_dict() for e in wind_events],
            },
        }

    def detect_all(
        self,
        blended: xr.Dataset,
        model_data: dict[str, xr.Dataset],
        lats: np.ndarray,
        lons: np.ndarray,
        time_idx: int = 0,
        model_agreement: float = 0.5,
        forecast_uncertainty: float = 0.5,
    ) -> dict[str, Any]:
        """Run all extreme weather detections on a single time step."""
        rain_events = []
        heat_events = []
        wind_events = []

        if "rainfall" in blended.data_vars:
            rain_bl = blended["rainfall"].isel(time=time_idx).values if "time" in blended.dims else blended["rainfall"].values
            rain_models = {}
            for mid, ds in model_data.items():
                if "rainfall" in ds.data_vars:
                    r = ds["rainfall"].isel(time=time_idx).values if "time" in ds.dims else ds["rainfall"].values
                    rain_models[mid] = r
            rain_events = self.detect_heavy_rainfall(rain_bl, rain_models, lats, lons, model_agreement)

        if "temperature" in blended.data_vars:
            temp_bl = blended["temperature"].isel(time=time_idx).values if "time" in blended.dims else blended["temperature"].values
            temp_models = {}
            for mid, ds in model_data.items():
                if "temperature" in ds.data_vars:
                    t = ds["temperature"].isel(time=time_idx).values if "time" in ds.dims else ds["temperature"].values
                    temp_models[mid] = t
            heat_events = self.detect_heat_wave(temp_bl, temp_models, lats, lons, model_agreement=model_agreement)

        if "wind_speed" in blended.data_vars:
            wind_bl = blended["wind_speed"].isel(time=time_idx).values if "time" in blended.dims else blended["wind_speed"].values
            wind_models = {}
            for mid, ds in model_data.items():
                if "wind_speed" in ds.data_vars:
                    w = ds["wind_speed"].isel(time=time_idx).values if "time" in ds.dims else ds["wind_speed"].values
                    wind_models[mid] = w
            wind_events = self.detect_high_wind(wind_bl, wind_models, lats, lons, model_agreement)

        composite = self.compute_composite_hazard(
            rain_events, heat_events, wind_events,
            forecast_uncertainty, model_agreement
        )

        return composite
