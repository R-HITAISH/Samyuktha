"""
SAMYUKTA — Verification & Evaluation Framework

Compares:
- Individual models
- Equal-weight ensemble
- Static skill-weighted ensemble
- SAMYUKTA (RSC-MoE adaptive blend)

Against observations, generating comprehensive verification tables and charts.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
import pandas as pd
from dataclasses import dataclass, field
from typing import Any
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Verification metrics for a single model/method."""
    method_name: str
    variable: str
    mae: float
    rmse: float
    bias: float
    correlation: float
    skill_score: float
    brier_score: float | None = None
    csi: float | None = None
    pod: float | None = None
    far: float | None = None
    sample_count: int = 0
    region: str = "all"
    lead_time: int = 0
    season: str = "all"
    regime: str = "all"

    def to_dict(self) -> dict:
        d = {
            "method_name": self.method_name,
            "variable": self.variable,
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "bias": round(self.bias, 4),
            "correlation": round(self.correlation, 4),
            "skill_score": round(self.skill_score, 4),
            "sample_count": self.sample_count,
            "region": self.region,
            "lead_time": self.lead_time,
            "season": self.season,
            "regime": self.regime,
        }
        for k in ["brier_score", "csi", "pod", "far"]:
            v = getattr(self, k)
            if v is not None:
                d[k] = round(v, 4)
        return d


class VerificationFramework:
    """
    Automated verification comparing all baselines against SAMYUKTA.

    Baselines:
    1. Best single model
    2. Equal-weight ensemble
    3. Static historical-skill-weighted ensemble
    4. SAMYUKTA (RSC-MoE adaptive)
    """

    def __init__(self):
        self.results: list[VerificationResult] = []

    def _calc_metrics(
        self,
        forecast: np.ndarray,
        observation: np.ndarray,
        variable: str,
    ) -> dict[str, float]:
        mask = ~(np.isnan(forecast) | np.isnan(observation))
        fc = forecast[mask]
        obs = observation[mask]

        if len(fc) < 2:
            return {"mae": 999, "rmse": 999, "bias": 0, "correlation": 0, "skill_score": 0}

        err = fc - obs
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))

        if np.std(fc) > 1e-8 and np.std(obs) > 1e-8:
            corr = float(np.corrcoef(fc, obs)[0, 1])
        else:
            corr = 0.0

        # Normalize for skill score
        norms = {"rainfall": 50.0, "temperature": 10.0, "wind_speed": 15.0}
        norm = norms.get(variable, 50.0)
        rmse_score = 1.0 - min(rmse / norm, 1.0)
        skill = float(np.clip(0.4 * rmse_score + 0.3 * max(corr, 0) + 0.3 * (1 - min(abs(bias) / 10, 1)), 0, 1))

        result = {"mae": mae, "rmse": rmse, "bias": bias, "correlation": corr, "skill_score": skill, "n": len(fc)}

        # Event-based for rainfall
        if variable == "rainfall":
            threshold = 64.5
            fc_ev = fc > threshold
            obs_ev = obs > threshold
            h = np.sum(fc_ev & obs_ev)
            m = np.sum(~fc_ev & obs_ev)
            f = np.sum(fc_ev & ~obs_ev)
            result["pod"] = float(h / max(h + m, 1))
            result["far"] = float(f / max(h + f, 1))
            result["csi"] = float(h / max(h + m + f, 1))

        return result

    def evaluate(
        self,
        model_forecasts: dict[str, np.ndarray],
        blended_forecast: np.ndarray,
        observation: np.ndarray,
        variable: str,
        skill_scores: dict[str, float] | None = None,
        lead_time: int = 0,
        season: str = "all",
        regime: str = "all",
        region: str = "all",
    ) -> list[VerificationResult]:
        """
        Run full verification comparing all methods.

        Returns list of VerificationResult for each method.
        """
        results = []

        # 1. Individual models
        best_model_skill = -1
        best_model_name = ""
        for model_id, fc in model_forecasts.items():
            m = self._calc_metrics(fc, observation, variable)
            vr = VerificationResult(
                method_name=model_id, variable=variable,
                mae=m["mae"], rmse=m["rmse"], bias=m["bias"],
                correlation=m["correlation"], skill_score=m["skill_score"],
                brier_score=m.get("brier_score"), csi=m.get("csi"),
                pod=m.get("pod"), far=m.get("far"),
                sample_count=m.get("n", 0),
                lead_time=lead_time, season=season, regime=regime, region=region,
            )
            results.append(vr)
            if m["skill_score"] > best_model_skill:
                best_model_skill = m["skill_score"]
                best_model_name = model_id

        # 2. Equal-weight ensemble
        fc_stack = np.stack(list(model_forecasts.values()))
        equal_blend = np.mean(fc_stack, axis=0)
        m = self._calc_metrics(equal_blend, observation, variable)
        results.append(VerificationResult(
            method_name="Equal Ensemble", variable=variable,
            mae=m["mae"], rmse=m["rmse"], bias=m["bias"],
            correlation=m["correlation"], skill_score=m["skill_score"],
            csi=m.get("csi"), pod=m.get("pod"), far=m.get("far"),
            sample_count=m.get("n", 0),
            lead_time=lead_time, season=season, regime=regime, region=region,
        ))

        # 3. Static skill-weighted ensemble
        if skill_scores:
            skill_arr = np.array([skill_scores.get(mid, 0.5) for mid in model_forecasts.keys()])
            weights = skill_arr / max(skill_arr.sum(), 1e-8)
            static_blend = np.sum(fc_stack * weights[:, None] if fc_stack.ndim == 2 else fc_stack * weights.reshape(-1, 1, 1), axis=0)
            m = self._calc_metrics(static_blend, observation, variable)
            results.append(VerificationResult(
                method_name="Static Skill-Weighted", variable=variable,
                mae=m["mae"], rmse=m["rmse"], bias=m["bias"],
                correlation=m["correlation"], skill_score=m["skill_score"],
                csi=m.get("csi"), pod=m.get("pod"), far=m.get("far"),
                sample_count=m.get("n", 0),
                lead_time=lead_time, season=season, regime=regime, region=region,
            ))

        # 4. SAMYUKTA adaptive blend
        m = self._calc_metrics(blended_forecast, observation, variable)
        results.append(VerificationResult(
            method_name="SAMYUKTA", variable=variable,
            mae=m["mae"], rmse=m["rmse"], bias=m["bias"],
            correlation=m["correlation"], skill_score=m["skill_score"],
            csi=m.get("csi"), pod=m.get("pod"), far=m.get("far"),
            sample_count=m.get("n", 0),
            lead_time=lead_time, season=season, regime=regime, region=region,
        ))

        self.results.extend(results)
        return results

    def get_improvement_table(self, variable: str = None) -> list[dict]:
        """Generate improvement comparison table."""
        filtered = self.results
        if variable:
            filtered = [r for r in filtered if r.variable == variable]

        table = []
        samyukta = [r for r in filtered if r.method_name == "SAMYUKTA"]
        if not samyukta:
            return []

        sam = samyukta[0]

        for r in filtered:
            improvement = {
                "method": r.method_name,
                "variable": r.variable,
                "mae": r.mae,
                "rmse": r.rmse,
                "bias": r.bias,
                "correlation": r.correlation,
                "skill_score": r.skill_score,
            }
            if r.method_name != "SAMYUKTA":
                improvement["mae_improvement_%"] = round((r.mae - sam.mae) / max(r.mae, 1e-8) * 100, 2)
                improvement["rmse_improvement_%"] = round((r.rmse - sam.rmse) / max(r.rmse, 1e-8) * 100, 2)
                improvement["skill_improvement_%"] = round((sam.skill_score - r.skill_score) / max(r.skill_score, 1e-8) * 100, 2)
            table.append(improvement)

        return table

    def get_results_df(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.results])

    def save(self, path: str) -> None:
        data = [r.to_dict() for r in self.results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class AblationStudy:
    """
    Measures contribution of each feature group by removing it and measuring impact.

    Features tested:
    - Weather regime information
    - Spatial information
    - Historical skill
    - Model disagreement
    - Lead-time features
    """

    def __init__(self, blending_engine, verification: VerificationFramework):
        self.engine = blending_engine
        self.verification = verification

    def run(
        self,
        model_forecasts: dict[str, np.ndarray],
        observation: np.ndarray,
        variable: str,
        full_blend: np.ndarray,
        skill_scores: dict[str, float],
        quality_scores: dict[str, float],
    ) -> list[dict]:
        """Run ablation experiments."""
        results = []
        model_ids = list(model_forecasts.keys())
        fc_stack = np.stack(list(model_forecasts.values()))

        # Full model result
        m_full = self.verification._calc_metrics(full_blend, observation, variable)
        results.append({
            "experiment": "Full SAMYUKTA",
            "rmse": m_full["rmse"],
            "mae": m_full["mae"],
            "skill_score": m_full["skill_score"],
            "removed_feature": "none",
        })

        # Without skill (use equal weights)
        no_skill_blend = np.mean(fc_stack, axis=0)
        m = self.verification._calc_metrics(no_skill_blend, observation, variable)
        results.append({
            "experiment": "Without Historical Skill",
            "rmse": m["rmse"],
            "mae": m["mae"],
            "skill_score": m["skill_score"],
            "removed_feature": "historical_skill",
            "impact": round(m["rmse"] - m_full["rmse"], 4),
        })

        # Without quality (set all quality to 1.0)
        equal_qual = {mid: 1.0 for mid in model_ids}
        skill_arr = np.array([skill_scores.get(mid, 0.5) for mid in model_ids])
        w = skill_arr / max(skill_arr.sum(), 1e-8)
        no_qual_blend = np.sum(fc_stack * w.reshape(-1, *([1] * (fc_stack.ndim - 1))), axis=0)
        m = self.verification._calc_metrics(no_qual_blend, observation, variable)
        results.append({
            "experiment": "Without Data Quality",
            "rmse": m["rmse"],
            "mae": m["mae"],
            "skill_score": m["skill_score"],
            "removed_feature": "data_quality",
            "impact": round(m["rmse"] - m_full["rmse"], 4),
        })

        return results
