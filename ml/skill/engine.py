"""
SAMYUKTA — Historical Skill Engine

Calculates and stores rolling historical forecast skill for every model,
disaggregated by region, grid cell, variable, lead time, season, and regime.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
import pandas as pd
from dataclasses import dataclass, field
from typing import Any
from collections import defaultdict
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class SkillMetrics:
    """Comprehensive skill metrics for a model in a specific context."""
    model_id: str
    variable: str
    region_id: str = "all"
    lead_time: int = 0
    season: str = "all"
    regime: str = "all"
    sample_count: int = 0
    mae: float = 0.0
    rmse: float = 0.0
    bias: float = 0.0
    correlation: float = 0.0
    brier_score: float | None = None
    csi: float | None = None
    pod: float | None = None
    far: float | None = None
    fss: float | None = None
    skill_score: float = 0.0  # composite 0-1
    last_updated: str = ""

    def to_dict(self) -> dict:
        d = {
            "model_id": self.model_id,
            "variable": self.variable,
            "region_id": self.region_id,
            "lead_time": self.lead_time,
            "season": self.season,
            "regime": self.regime,
            "sample_count": self.sample_count,
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "bias": round(self.bias, 4),
            "correlation": round(self.correlation, 4),
            "skill_score": round(self.skill_score, 4),
            "last_updated": self.last_updated,
        }
        if self.brier_score is not None:
            d["brier_score"] = round(self.brier_score, 4)
        if self.csi is not None:
            d["csi"] = round(self.csi, 4)
        if self.pod is not None:
            d["pod"] = round(self.pod, 4)
        if self.far is not None:
            d["far"] = round(self.far, 4)
        return d


class HistoricalSkillEngine:
    """
    Calculates and stores rolling skill metrics for all models.

    Key principle: skill is calculated separately by
    - region
    - variable
    - lead time
    - season
    - weather regime
    so the blending engine can select the best model for each specific context.
    """

    def __init__(self, window_days: int = 30):
        self.window_days = window_days
        self.skill_db: dict[str, SkillMetrics] = {}  # key -> SkillMetrics

    def _make_key(self, model_id: str, variable: str, region: str,
                  lead_time: int, season: str, regime: str) -> str:
        return f"{model_id}|{variable}|{region}|{lead_time}|{season}|{regime}"

    def calculate_metrics(
        self,
        forecast: np.ndarray,
        observation: np.ndarray,
        variable: str,
        threshold: float = None,
    ) -> dict[str, float]:
        """Calculate comprehensive metrics for a forecast-observation pair."""
        # Remove NaNs
        mask = ~(np.isnan(forecast) | np.isnan(observation))
        fc = forecast[mask]
        obs = observation[mask]

        if len(fc) < 2:
            return {"mae": 999, "rmse": 999, "bias": 0, "correlation": 0, "skill_score": 0}

        errors = fc - obs

        metrics = {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "bias": float(np.mean(errors)),
            "sample_count": int(len(fc)),
        }

        # Correlation
        if np.std(fc) > 1e-8 and np.std(obs) > 1e-8:
            metrics["correlation"] = float(np.corrcoef(fc, obs)[0, 1])
        else:
            metrics["correlation"] = 0.0

        # Event-based metrics for rainfall
        if variable == "rainfall" and threshold is not None:
            fc_event = fc > threshold
            obs_event = obs > threshold

            hits = np.sum(fc_event & obs_event)
            misses = np.sum(~fc_event & obs_event)
            false_alarms = np.sum(fc_event & ~obs_event)
            correct_negs = np.sum(~fc_event & ~obs_event)

            # POD
            pod = hits / max(hits + misses, 1)
            metrics["pod"] = float(pod)

            # FAR
            far = false_alarms / max(hits + false_alarms, 1)
            metrics["far"] = float(far)

            # CSI (Critical Success Index)
            csi = hits / max(hits + misses + false_alarms, 1)
            metrics["csi"] = float(csi)

            # Brier Score
            prob_fc = np.clip(fc / max(np.max(fc), 1), 0, 1)
            obs_binary = obs_event.astype(float)
            metrics["brier_score"] = float(np.mean((prob_fc - obs_binary) ** 2))

        # Composite skill score (0 = worst, 1 = best)
        # Normalize RMSE to [0,1] range
        if variable == "rainfall":
            rmse_norm = 1.0 - min(metrics["rmse"] / 50.0, 1.0)
        elif variable == "temperature":
            rmse_norm = 1.0 - min(metrics["rmse"] / 10.0, 1.0)
        elif variable == "wind_speed":
            rmse_norm = 1.0 - min(metrics["rmse"] / 15.0, 1.0)
        else:
            rmse_norm = 1.0 - min(metrics["rmse"] / 50.0, 1.0)

        corr_score = max(metrics["correlation"], 0)
        bias_score = 1.0 - min(abs(metrics["bias"]) / 10.0, 1.0)

        metrics["skill_score"] = float(np.clip(
            0.4 * rmse_norm + 0.3 * corr_score + 0.3 * bias_score,
            0, 1
        ))

        return metrics

    def update_skill(
        self,
        model_id: str,
        forecast: xr.Dataset,
        observation: xr.Dataset,
        variable: str,
        region_id: str = "all",
        lead_time: int = 0,
        season: str = "all",
        regime: str = "all",
        threshold: float = None,
    ) -> SkillMetrics:
        """
        Calculate and store skill metrics for a specific model/context combination.
        Uses exponential weighting so recent performance matters more.
        """
        fc_vals = forecast[variable].values.ravel()
        obs_vals = observation[variable].values.ravel()

        metrics = self.calculate_metrics(fc_vals, obs_vals, variable, threshold)

        key = self._make_key(model_id, variable, region_id, lead_time, season, regime)

        skill = SkillMetrics(
            model_id=model_id,
            variable=variable,
            region_id=region_id,
            lead_time=lead_time,
            season=season,
            regime=regime,
            sample_count=metrics.get("sample_count", 0),
            mae=metrics["mae"],
            rmse=metrics["rmse"],
            bias=metrics["bias"],
            correlation=metrics["correlation"],
            brier_score=metrics.get("brier_score"),
            csi=metrics.get("csi"),
            pod=metrics.get("pod"),
            far=metrics.get("far"),
            skill_score=metrics["skill_score"],
            last_updated=pd.Timestamp.utcnow().isoformat(),
        )

        # Exponential rolling: blend with existing if present
        if key in self.skill_db:
            old = self.skill_db[key]
            alpha = 0.3  # weight for new data
            skill.mae = alpha * skill.mae + (1 - alpha) * old.mae
            skill.rmse = alpha * skill.rmse + (1 - alpha) * old.rmse
            skill.bias = alpha * skill.bias + (1 - alpha) * old.bias
            skill.correlation = alpha * skill.correlation + (1 - alpha) * old.correlation
            skill.skill_score = alpha * skill.skill_score + (1 - alpha) * old.skill_score
            skill.sample_count += old.sample_count

        self.skill_db[key] = skill
        return skill

    def batch_evaluate(
        self,
        forecasts: dict[str, xr.Dataset],
        observation: xr.Dataset,
        variables: list[str],
        lead_time: int = 0,
        season: str = "all",
        regime: str = "all",
        regions: list[dict] | None = None,
    ) -> dict[str, list[SkillMetrics]]:
        """Evaluate all models against observations."""
        all_results: dict[str, list[SkillMetrics]] = {}

        for model_id, fc_ds in forecasts.items():
            results = []
            for var in variables:
                if var not in fc_ds.data_vars or var not in observation.data_vars:
                    continue

                # Overall skill
                threshold = 64.5 if var == "rainfall" else None
                skill = self.update_skill(
                    model_id, fc_ds, observation, var,
                    lead_time=lead_time, season=season, regime=regime,
                    threshold=threshold,
                )
                results.append(skill)

                # Regional skill
                if regions:
                    for region in regions:
                        bbox = region["bbox"]
                        fc_reg = fc_ds.sel(
                            lat=slice(bbox[0], bbox[2]),
                            lon=slice(bbox[1], bbox[3])
                        )
                        obs_reg = observation.sel(
                            lat=slice(bbox[0], bbox[2]),
                            lon=slice(bbox[1], bbox[3])
                        )
                        if fc_reg[var].size > 0 and obs_reg[var].size > 0:
                            skill_reg = self.update_skill(
                                model_id, fc_reg, obs_reg, var,
                                region_id=region["id"],
                                lead_time=lead_time, season=season, regime=regime,
                                threshold=threshold,
                            )
                            results.append(skill_reg)

            all_results[model_id] = results

        return all_results

    def get_skill(
        self,
        model_id: str,
        variable: str,
        region_id: str = "all",
        lead_time: int = 0,
        season: str = "all",
        regime: str = "all",
    ) -> SkillMetrics | None:
        """Retrieve stored skill for a specific context."""
        key = self._make_key(model_id, variable, region_id, lead_time, season, regime)
        return self.skill_db.get(key)

    def get_skill_vector(
        self,
        model_ids: list[str],
        variable: str,
        region_id: str = "all",
        lead_time: int = 0,
        season: str = "all",
        regime: str = "all",
    ) -> np.ndarray:
        """Get skill scores as a vector for all models (for blending engine)."""
        scores = []
        for mid in model_ids:
            skill = self.get_skill(mid, variable, region_id, lead_time, season, regime)
            scores.append(skill.skill_score if skill else 0.5)  # default neutral
        return np.array(scores)

    def get_all_skills(self) -> list[dict]:
        """Export all stored skills."""
        return [s.to_dict() for s in self.skill_db.values()]

    def get_model_comparison(self, variable: str, lead_time: int = 0) -> list[dict]:
        """Compare all models for a specific variable and lead time."""
        results = []
        seen_models = set()
        for key, skill in self.skill_db.items():
            if skill.variable == variable and skill.lead_time == lead_time and skill.region_id == "all":
                if skill.model_id not in seen_models:
                    results.append(skill.to_dict())
                    seen_models.add(skill.model_id)
        return sorted(results, key=lambda x: x["skill_score"], reverse=True)

    def save(self, path: str) -> None:
        data = {k: v.to_dict() for k, v in self.skill_db.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str, window_days: int = 30) -> "HistoricalSkillEngine":
        engine = cls(window_days=window_days)
        with open(path) as f:
            data = json.load(f)
        for key, d in data.items():
            engine.skill_db[key] = SkillMetrics(**{
                k: v for k, v in d.items() if k in SkillMetrics.__dataclass_fields__
            })
        return engine
