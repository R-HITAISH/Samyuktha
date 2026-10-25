"""
SAMYUKTA — Master Forecast Pipeline Orchestrator

Executes the complete end-to-end pipeline:
INGEST → VALIDATE → REGRID → ALIGN → QC → BIAS CORRECT → REGIME DETECT →
SKILL LOOKUP → META-LEARNER → DYNAMIC WEIGHTS → BLEND → UNCERTAINTY →
EXTREME EVENTS → VERIFY → API
"""
from __future__ import annotations

import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
import logging
import json
import time
import uuid

logger = logging.getLogger(__name__)


@dataclass
class ForecastRun:
    """Complete audit record of a single forecast pipeline run."""
    run_id: str
    timestamp: str
    init_time: str
    lead_time_hours: int
    variable: str
    models_used: list[str]
    models_unavailable: list[str]
    regime_detected: str
    regime_confidence: float
    model_weights: dict[str, float]
    blended_forecast_mean: float
    uncertainty: float
    confidence: float
    model_agreement: float
    quality_scores: dict[str, float]
    skill_scores: dict[str, float]
    hazard_events: list[dict]
    verification: dict | None = None
    explanation: str = ""
    data_mode: str = "DEMO"
    processing_time_ms: float = 0
    counterfactuals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "init_time": self.init_time,
            "lead_time_hours": self.lead_time_hours,
            "variable": self.variable,
            "models_used": self.models_used,
            "models_unavailable": self.models_unavailable,
            "regime_detected": self.regime_detected,
            "regime_confidence": round(self.regime_confidence, 4),
            "model_weights": {k: round(v, 4) for k, v in self.model_weights.items()},
            "blended_forecast_mean": round(self.blended_forecast_mean, 2),
            "uncertainty": round(self.uncertainty, 4),
            "confidence": round(self.confidence, 4),
            "model_agreement": round(self.model_agreement, 4),
            "quality_scores": {k: round(v, 4) for k, v in self.quality_scores.items()},
            "skill_scores": {k: round(v, 4) for k, v in self.skill_scores.items()},
            "hazard_events": self.hazard_events,
            "verification": self.verification,
            "explanation": self.explanation,
            "data_mode": self.data_mode,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "counterfactuals": self.counterfactuals,
        }


class SamyuktaPipeline:
    """
    Master pipeline that orchestrates the complete SAMYUKTA workflow.

    This is the main entry point that connects all ML components.
    """

    def __init__(self, data_dir: str = "data/demo", config: dict = None):
        self.data_dir = Path(data_dir)
        self.config = config or {}

        # Import components
        from ml.preprocessing.pipeline import DataQualityEngine, BiasCorrector, PhysicsValidator
        from ml.regimes.detector import WeatherRegimeDetector
        from ml.skill.engine import HistoricalSkillEngine
        from ml.blending.rsc_moe import AdaptiveBlendingEngine
        from ml.extremes.engine import ExtremeWeatherEngine
        from ml.evaluation.verification import VerificationFramework
        from ml.calibration import CalibrationEngine
        from ml.uncertainty.engine import UncertaintyEngine

        self.model_ids = ["ncum_g", "ncum_r", "gfs", "wrf", "ai_model"]
        self.variables = ["rainfall", "temperature", "wind_speed"]

        # Initialize components
        self.quality_engine = DataQualityEngine()
        self.bias_corrector = BiasCorrector()
        self.regime_detector = WeatherRegimeDetector(n_regimes=8)
        self.skill_engine = HistoricalSkillEngine(window_days=30)
        self.blending_engine = AdaptiveBlendingEngine(model_ids=self.model_ids)
        self.extreme_engine = ExtremeWeatherEngine()
        self.verification = VerificationFramework()
        self.physics = PhysicsValidator()
        self.calibration_engine = CalibrationEngine()
        self.uncertainty_engine = UncertaintyEngine(confidence_level=0.90)

        # State
        self.is_initialized = False
        self.forecast_runs: list[ForecastRun] = []
        self.ground_truth: xr.Dataset | None = None
        self.model_forecasts: dict[str, dict[int, xr.Dataset]] = {}  # model -> lt -> dataset
        self.lats: np.ndarray | None = None
        self.lons: np.ndarray | None = None

    def initialize(self) -> dict:
        """Load demo data and fit components."""
        start = time.time()
        status = {"steps": [], "errors": []}

        try:
            # 1. Load ground truth
            gt_path = self.data_dir / "ground_truth.nc"
            if gt_path.exists():
                self.ground_truth = xr.open_dataset(gt_path)
                self.lats = self.ground_truth.lat.values
                self.lons = self.ground_truth.lon.values
                status["steps"].append("Ground truth loaded")
            else:
                status["errors"].append(f"Ground truth not found: {gt_path}")
                return status

            # 2. Load model forecasts
            for model_id in self.model_ids:
                self.model_forecasts[model_id] = {}
                for lt in [6, 12, 24, 48, 72, 120]:
                    fc_path = self.data_dir / f"forecast_{model_id}_lt{lt:03d}.nc"
                    if fc_path.exists():
                        self.model_forecasts[model_id][lt] = xr.open_dataset(fc_path)
                status["steps"].append(f"Loaded {model_id} forecasts")

            # 3. Fit regime detector
            self.regime_detector.fit(self.ground_truth)
            status["steps"].append("Regime detector fitted")

            # 4. Calculate historical skill
            for model_id in self.model_ids:
                for lt, fc_ds in self.model_forecasts.get(model_id, {}).items():
                    for var in self.variables:
                        if var in fc_ds.data_vars and var in self.ground_truth.data_vars:
                            self.skill_engine.update_skill(
                                model_id, fc_ds, self.ground_truth, var,
                                lead_time=lt,
                            )
            status["steps"].append("Historical skill calculated")

            # 5. Fit bias corrector
            for model_id in self.model_ids:
                lt24 = self.model_forecasts.get(model_id, {}).get(24)
                if lt24 is not None:
                    self.bias_corrector.fit(model_id, lt24, self.ground_truth)
            status["steps"].append("Bias corrector fitted")

            # 6. Fit calibration & conformal uncertainty engine
            for var in self.variables:
                threshold = 64.5 if var == "rainfall" else (40.0 if var == "temperature" else 17.2)
                lt24_list = [
                    self.model_forecasts[m][24][var].values
                    for m in self.model_ids
                    if 24 in self.model_forecasts.get(m, {}) and var in self.model_forecasts[m][24].data_vars
                ]
                if lt24_list and var in self.ground_truth.data_vars:
                    ens_mean = np.mean(lt24_list, axis=0)
                    gt_arr = self.ground_truth[var].values
                    self.calibration_engine.fit_variable(var, ens_mean, gt_arr, threshold=threshold)
                    residuals = ens_mean - gt_arr
                    self.uncertainty_engine.calibrate_conformal(residuals.ravel(), var)
            status["steps"].append("Calibration & Uncertainty engines fitted")

            self.is_initialized = True
            status["time_ms"] = round((time.time() - start) * 1000, 2)
            status["steps"].append(f"Pipeline initialized in {status['time_ms']}ms")

        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            status["errors"].append(str(e))

        return status

    def run_forecast(
        self,
        time_idx: int = 0,
        lead_time: int = 24,
        variable: str = "rainfall",
    ) -> ForecastRun:
        """Execute complete forecast pipeline for one time step."""
        start = time.time()
        run_id = str(uuid.uuid4())[:8]

        if not self.is_initialized:
            self.initialize()

        # 1. Collect model forecasts for this time/lead
        available_models = []
        unavailable_models = []
        fc_arrays = {}
        fc_datasets = {}
        quality_scores = {}

        for model_id in self.model_ids:
            fc_ds = self.model_forecasts.get(model_id, {}).get(lead_time)
            if fc_ds is not None and variable in fc_ds.data_vars:
                n_time = fc_ds.dims.get("time", 0)
                if time_idx < n_time:
                    # Quality check
                    qr = self.quality_engine.assess(fc_ds, model_id)
                    quality_scores[model_id] = qr.quality_score

                    # Bias correction
                    fc_corrected = self.bias_corrector.correct(model_id, fc_ds)

                    arr = fc_corrected[variable].isel(time=time_idx).values
                    fc_arrays[model_id] = arr
                    fc_datasets[model_id] = fc_corrected
                    available_models.append(model_id)
                else:
                    unavailable_models.append(model_id)
            else:
                unavailable_models.append(model_id)

        if not available_models:
            return ForecastRun(
                run_id=run_id,
                timestamp=datetime.utcnow().isoformat(),
                init_time="", lead_time_hours=lead_time, variable=variable,
                models_used=[], models_unavailable=self.model_ids,
                regime_detected="unknown", regime_confidence=0,
                model_weights={}, blended_forecast_mean=0,
                uncertainty=999, confidence=0, model_agreement=0,
                quality_scores={}, skill_scores={},
                hazard_events=[], explanation="No models available",
                data_mode="DEMO",
            )

        # 2. Detect weather regime
        regime = self.regime_detector.predict(self.ground_truth, time_idx=time_idx)

        # 3. Get historical skill
        skill_scores = {}
        for mid in available_models:
            skill = self.skill_engine.get_skill(mid, variable, lead_time=lead_time)
            skill_scores[mid] = skill.skill_score if skill else 0.5

        # 4. Spatial blending
        blend_result = self.blending_engine.blend_spatial(
            fc_arrays, quality_scores=quality_scores,
            regime_id=regime.regime_id, lead_time_hours=lead_time,
            lats=self.lats, lons=self.lons,
        )

        blended_grid = blend_result["blended"]

        # Calibrate forecast field with Quantile Mapping
        blended_grid = self.calibration_engine.calibrate_forecast(blended_grid, variable)

        # Apply physics constraints
        if variable == "rainfall":
            blended_grid = np.clip(blended_grid, 0, None)
        elif variable == "wind_speed":
            blended_grid = np.clip(blended_grid, 0, None)

        # 5. Single-point blend for summary
        point_forecasts = {mid: float(np.nanmean(arr)) for mid, arr in fc_arrays.items()}
        point_blend = self.blending_engine.blend(
            point_forecasts, skill_scores, quality_scores,
            regime_id=regime.regime_id, lead_time_hours=lead_time,
        )

        # Quantify calibrated uncertainty using UncertaintyEngine
        fc_vals = np.array(list(point_forecasts.values()))
        weight_vals = np.array([point_blend["model_weights"].get(m, 0.0) for m in point_forecasts.keys()])
        weight_vals = weight_vals / max(np.sum(weight_vals), 1e-8)
        unc_est = self.uncertainty_engine.from_ensemble_spread(fc_vals, weights=weight_vals, variable=variable)
        point_blend["uncertainty"] = unc_est.spread
        point_blend["confidence"] = float(np.clip(1.0 - unc_est.uncertainty_score, 0.05, 0.99))

        # 6. Extreme event detection
        blended_ds = xr.Dataset(
            {variable: (["lat", "lon"], blended_grid)},
            coords={"lat": self.lats, "lon": self.lons},
        )
        # Build model data dict for extreme detection
        model_ds_dict = {}
        for mid, arr in fc_arrays.items():
            model_ds_dict[mid] = xr.Dataset(
                {variable: (["lat", "lon"], arr)},
                coords={"lat": self.lats, "lon": self.lons},
            )

        hazard_result = self.extreme_engine.detect_all(
            blended_ds, model_ds_dict, self.lats, self.lons,
            time_idx=0,
            model_agreement=point_blend["model_agreement"],
            forecast_uncertainty=point_blend["uncertainty"],
        )

        # 7. Verification against ground truth
        obs_arr = self.ground_truth[variable].isel(time=time_idx).values
        verification_results = self.verification.evaluate(
            {mid: arr.ravel() for mid, arr in fc_arrays.items()},
            blended_grid.ravel(),
            obs_arr.ravel(),
            variable,
            skill_scores=skill_scores,
            lead_time=lead_time,
        )

        # 8. Generate explanation
        explanation = self.blending_engine.generate_explanation(point_blend)

        # Build run record
        processing_time = (time.time() - start) * 1000

        run = ForecastRun(
            run_id=run_id,
            timestamp=datetime.utcnow().isoformat(),
            init_time=str(self.ground_truth.time.values[time_idx]),
            lead_time_hours=lead_time,
            variable=variable,
            models_used=available_models,
            models_unavailable=unavailable_models,
            regime_detected=regime.regime_id,
            regime_confidence=regime.confidence,
            model_weights=point_blend["model_weights"],
            blended_forecast_mean=point_blend["blended_forecast"],
            uncertainty=point_blend["uncertainty"],
            confidence=point_blend["confidence"],
            model_agreement=point_blend["model_agreement"],
            quality_scores=quality_scores,
            skill_scores=skill_scores,
            hazard_events=hazard_result.get("all_events", {}),
            verification={r.method_name: r.to_dict() for r in verification_results},
            explanation=explanation,
            data_mode="DEMO / SYNTHETIC DATA",
            processing_time_ms=processing_time,
            counterfactuals=point_blend.get("counterfactuals", {}),
        )

        self.forecast_runs.append(run)
        logger.info(f"Forecast run {run_id} completed in {processing_time:.1f}ms")

        return run

    def run_demo_scenario(self) -> dict:
        """
        Execute the complete SIH demo scenario.

        Simulates a major rainfall event approaching India:
        1. Load multi-model forecasts
        2. Show disagreement
        3. Detect regime
        4. Calculate skill
        5. Generate adaptive weights
        6. Produce unified forecast
        7. Calculate uncertainty
        8. Detect extreme weather risk
        9. Show baseline vs SAMYUKTA
        """
        logger.info("Starting SAMYUKTA demo scenario...")
        demo_result = {
            "scenario": "Heavy Rainfall Event — Indian Region",
            "steps": [],
            "data_mode": "DEMO / SYNTHETIC DATA",
        }

        # Initialize
        init_status = self.initialize()
        demo_result["initialization"] = init_status

        if not self.is_initialized:
            demo_result["error"] = "Failed to initialize pipeline"
            return demo_result

        # Find a time with interesting weather (high rainfall)
        rain_means = []
        n_times = self.ground_truth.dims.get("time", 0)
        for t in range(min(n_times, 90)):
            rain = self.ground_truth["rainfall"].isel(time=t).values
            rain_means.append(float(np.nanmean(rain)))

        # Pick a day with high rainfall for demo
        interesting_idx = int(np.argmax(rain_means[:min(len(rain_means), 90)]))

        demo_result["steps"].append({
            "step": 1,
            "name": "Select High-Impact Day",
            "time_idx": interesting_idx,
            "rainfall_mean": rain_means[interesting_idx],
        })

        # Run for multiple variables
        runs = {}
        for var in self.variables:
            for lt in [24, 48, 72]:
                run = self.run_forecast(
                    time_idx=interesting_idx,
                    lead_time=lt,
                    variable=var,
                )
                runs[f"{var}_lt{lt}"] = run.to_dict()

        demo_result["forecast_runs"] = runs

        # Verification summary
        improvement_table = self.verification.get_improvement_table("rainfall")
        demo_result["improvement_table"] = improvement_table

        # Weight maps
        if "rainfall_lt24" in runs:
            demo_result["primary_run"] = runs["rainfall_lt24"]

        demo_result["steps"].extend([
            {"step": 2, "name": "Multi-Model Forecasts Loaded", "models": self.model_ids},
            {"step": 3, "name": "Regime Detection", "regime": runs.get("rainfall_lt24", {}).get("regime_detected", "unknown")},
            {"step": 4, "name": "Historical Skill Retrieved"},
            {"step": 5, "name": "Adaptive Weights Generated"},
            {"step": 6, "name": "Forecast Blended"},
            {"step": 7, "name": "Uncertainty Quantified"},
            {"step": 8, "name": "Extreme Events Detected"},
            {"step": 9, "name": "Verification Completed"},
        ])

        logger.info("Demo scenario completed successfully")
        return demo_result

    def get_weight_map_data(
        self,
        time_idx: int = 0,
        lead_time: int = 24,
        variable: str = "rainfall",
    ) -> dict:
        """Get weight map data for visualization."""
        fc_arrays = {}
        quality_scores = {}

        for model_id in self.model_ids:
            fc_ds = self.model_forecasts.get(model_id, {}).get(lead_time)
            if fc_ds is not None and variable in fc_ds.data_vars:
                n_time = fc_ds.dims.get("time", 0)
                if time_idx < n_time:
                    fc_corrected = self.bias_corrector.correct(model_id, fc_ds)
                    fc_arrays[model_id] = fc_corrected[variable].isel(time=time_idx).values
                    qr = self.quality_engine.assess(fc_ds, model_id)
                    quality_scores[model_id] = qr.quality_score

        if not fc_arrays:
            return {}

        regime = self.regime_detector.predict(self.ground_truth, time_idx=time_idx)

        result = self.blending_engine.blend_spatial(
            fc_arrays, quality_scores=quality_scores,
            regime_id=regime.regime_id, lead_time_hours=lead_time,
            lats=self.lats, lons=self.lons,
        )

        # Convert to serializable format
        output = {
            "lats": self.lats.tolist(),
            "lons": self.lons.tolist(),
            "blended": result["blended"].tolist(),
            "uncertainty": result["uncertainty"].tolist(),
            "weight_maps": {
                mid: wm.tolist() for mid, wm in result["weight_maps"].items()
            },
            "regime": regime.regime_id,
            "lead_time": lead_time,
            "variable": variable,
        }
        return output

    def get_system_status(self) -> dict:
        """Get current system health status."""
        model_status = {}
        for mid in self.model_ids:
            has_data = mid in self.model_forecasts and len(self.model_forecasts[mid]) > 0
            model_status[mid] = {
                "available": has_data,
                "lead_times": list(self.model_forecasts.get(mid, {}).keys()),
            }

        return {
            "initialized": self.is_initialized,
            "data_mode": "DEMO / SYNTHETIC DATA",
            "models": model_status,
            "total_runs": len(self.forecast_runs),
            "last_run": self.forecast_runs[-1].to_dict() if self.forecast_runs else None,
            "ground_truth_available": self.ground_truth is not None,
            "grid_shape": list(self.ground_truth.rainfall.shape) if self.ground_truth is not None else None,
        }
