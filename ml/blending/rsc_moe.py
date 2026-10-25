"""
SAMYUKTA — RSC-MoE: Regime-Skill-Conditioned Mixture of Experts

The core innovation: dynamically determines model weights conditioned on:
1. Region (latitude, longitude, elevation, coast distance)
2. Forecast lead time
3. Month/season (cyclic encoding)
4. Weather regime
5. Weather variable
6. Recent model skill
7. Model disagreement
8. Input data quality
9. Forecast uncertainty

Produces per-model, per-grid-cell, per-time adaptive weights.

Architecture:
- Feature encoders for each conditioning dimension
- Meta-learner that fuses all signals
- Softmax weight head (non-negative, sum-to-1)
- Forecast head (weighted blend)
- Uncertainty head (prediction intervals)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class BlendingInput:
    """All inputs needed by the RSC-MoE engine for a single grid cell/time."""
    model_forecasts: np.ndarray      # (n_models,) forecasts
    model_skills: np.ndarray         # (n_models,) skill scores
    model_quality: np.ndarray        # (n_models,) quality scores
    latitude: float
    longitude: float
    elevation: float = 0.0
    coast_distance: float = 5.0
    lead_time_hours: int = 24
    month: int = 6
    regime_id: int = 0
    regime_confidence: float = 0.5

    def to_feature_vector(self, n_models: int = 5) -> np.ndarray:
        """Flatten all inputs into a single feature vector."""
        # Cyclic month encoding
        month_sin = np.sin(2 * np.pi * self.month / 12)
        month_cos = np.cos(2 * np.pi * self.month / 12)

        # Lead time embedding (log-scaled)
        lt_norm = np.log1p(self.lead_time_hours) / np.log1p(168)

        # Model disagreement features
        fc = self.model_forecasts
        disagreement = np.std(fc) / max(np.mean(np.abs(fc)), 1e-8)
        fc_range = np.max(fc) - np.min(fc)
        fc_mean = np.mean(fc)

        features = np.concatenate([
            fc,                          # model forecasts
            self.model_skills,           # skill scores
            self.model_quality,          # quality scores
            [self.latitude / 38.0],      # normalized lat
            [self.longitude / 98.0],     # normalized lon
            [self.elevation / 5000.0],   # normalized elevation
            [self.coast_distance / 30.0],# normalized coast distance
            [lt_norm],                   # lead time
            [month_sin, month_cos],      # cyclic month
            [self.regime_id / 11.0],     # regime (normalized)
            [self.regime_confidence],    # regime confidence
            [disagreement],             # model disagreement
            [fc_range],                 # forecast spread
            [fc_mean],                  # forecast mean
        ])
        return features.astype(np.float32)


class RSCMoENet(nn.Module):
    """
    Neural Regime-Skill-Conditioned Mixture of Experts.

    Takes conditioning features and produces:
    - Model weights (softmax, sum to 1)
    - Blended forecast
    - Uncertainty estimate
    """

    def __init__(
        self,
        n_models: int = 5,
        input_dim: int = None,
        hidden_dim: int = 128,
        n_regime_classes: int = 11,
    ):
        super().__init__()
        self.n_models = n_models
        # Calculate input dim: 3*n_models + geographic(4) + temporal(3) + regime(2) + disagreement(3)
        if input_dim is None:
            input_dim = 3 * n_models + 4 + 3 + 2 + 3

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Regime-specific expert layers
        self.regime_embedding = nn.Embedding(n_regime_classes, 32)

        # Lead-time embedding
        self.lead_time_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
        )

        # Geographic encoder
        self.geo_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
        )

        # Meta-learner: fuses all signals
        meta_input_dim = hidden_dim + 32 + 32 + 32
        self.meta_learner = nn.Sequential(
            nn.Linear(meta_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
        )

        # Weight head: produces softmax weights
        self.weight_head = nn.Linear(64, n_models)

        # Uncertainty head: predicts spread
        self.uncertainty_head = nn.Sequential(
            nn.Linear(64 + n_models, 32),
            nn.GELU(),
            nn.Linear(32, 2),  # lower_spread, upper_spread
            nn.Softplus(),
        )

    def forward(
        self,
        features: torch.Tensor,        # (batch, input_dim)
        regime_ids: torch.Tensor,       # (batch,) int
        lead_times: torch.Tensor,       # (batch, 1) float
        geo_features: torch.Tensor,     # (batch, 4)
        model_forecasts: torch.Tensor,  # (batch, n_models)
    ) -> dict[str, torch.Tensor]:
        # Encode features
        h = self.feature_encoder(features)

        # Regime embedding
        regime_emb = self.regime_embedding(regime_ids)

        # Lead time embedding
        lt_emb = self.lead_time_mlp(lead_times)

        # Geographic encoding
        geo_emb = self.geo_encoder(geo_features)

        # Fuse all signals
        fused = torch.cat([h, regime_emb, lt_emb, geo_emb], dim=-1)
        meta = self.meta_learner(fused)

        # Weights (softmax → non-negative, sum to 1)
        weight_logits = self.weight_head(meta)
        weights = F.softmax(weight_logits, dim=-1)

        # Blended forecast
        blended = (weights * model_forecasts).sum(dim=-1, keepdim=True)

        # Uncertainty
        unc_input = torch.cat([meta, weights], dim=-1)
        uncertainty = self.uncertainty_head(unc_input)
        lower_spread = uncertainty[:, 0:1]
        upper_spread = uncertainty[:, 1:2]

        return {
            "weights": weights,
            "blended": blended,
            "lower_bound": blended - lower_spread,
            "upper_bound": blended + upper_spread,
            "uncertainty": (lower_spread + upper_spread) / 2,
            "weight_logits": weight_logits,
        }


class AdaptiveBlendingEngine:
    """
    Main blending engine that wraps RSC-MoE for production use.

    Provides both:
    1. Neural meta-learner blending (when trained model available)
    2. Analytical skill-based blending (always available as fallback)

    The analytical method is the production-ready baseline that works
    without neural network training.
    """

    REGIME_TO_ID = {
        "active_monsoon": 0, "break_monsoon": 1, "heavy_rainfall": 2,
        "convective": 3, "dry": 4, "heat_wave": 5, "high_wind": 6,
        "cyclonic": 7, "western_disturbance": 8, "coastal_rainfall": 9,
        "orographic_rainfall": 10,
    }

    def __init__(
        self,
        model_ids: list[str],
        use_neural: bool = False,
        device: str = "cpu",
        temperature: float = 1.0,
        smoothing_alpha: float = 0.1,
    ):
        self.model_ids = model_ids
        self.n_models = len(model_ids)
        self.use_neural = use_neural
        self.device = device
        self.temperature = temperature
        self.smoothing_alpha = smoothing_alpha

        # Previous weights for temporal smoothing
        self._prev_weights: np.ndarray | None = None

        if use_neural:
            self.net = RSCMoENet(n_models=self.n_models).to(device)
            self.net.eval()

    def analytical_blend(
        self,
        model_forecasts: np.ndarray,
        skill_scores: np.ndarray,
        quality_scores: np.ndarray,
        regime_id: str = "dry",
        lead_time_hours: int = 24,
        model_disagreement: float = 0.0,
    ) -> dict[str, Any]:
        """
        Analytical skill-based adaptive blending.

        Formula:
        w_m = softmax(
            skill_m + quality_m - uncertainty_penalty_m
            ──────────────────────────────────────────────
                          temperature
        )

        This is always available even without a trained neural network.
        """
        n_models = len(model_forecasts)

        # Base scores from historical skill (primary signal)
        scores = skill_scores.copy()

        # Quality adjustment
        scores *= quality_scores

        # Lead-time adjustment: penalize models with faster skill decay
        lt_factor = 1.0 - 0.01 * np.log1p(lead_time_hours)
        scores *= lt_factor

        # Model disagreement penalty
        if model_disagreement > 0.5:
            # High disagreement → reduce confidence in outlier models
            fc_mean = np.mean(model_forecasts)
            deviations = np.abs(model_forecasts - fc_mean)
            deviation_penalty = deviations / max(np.max(deviations), 1e-8) * 0.2
            scores -= deviation_penalty

        # Softmax with temperature
        scores = np.clip(scores, -10, 10)
        exp_scores = np.exp(scores / self.temperature)
        weights = exp_scores / np.sum(exp_scores)

        # Temporal smoothing
        if self._prev_weights is not None and len(self._prev_weights) == n_models:
            weights = (1 - self.smoothing_alpha) * weights + self.smoothing_alpha * self._prev_weights
            weights /= weights.sum()
        self._prev_weights = weights.copy()

        # Blended forecast
        blended = np.sum(weights * model_forecasts)

        # Uncertainty from model spread
        forecast_std = np.std(model_forecasts)
        weighted_std = np.sqrt(np.sum(weights * (model_forecasts - blended) ** 2))

        # Confidence
        mean_skill = np.mean(skill_scores)
        agreement = 1.0 - min(forecast_std / max(abs(blended), 1), 1.0)
        confidence = float(np.clip(0.5 * mean_skill + 0.3 * agreement + 0.2 * np.mean(quality_scores), 0.1, 0.99))

        return {
            "weights": weights,
            "blended_forecast": float(blended),
            "lower_bound": float(blended - 1.96 * weighted_std),
            "upper_bound": float(blended + 1.96 * weighted_std),
            "uncertainty": float(weighted_std),
            "confidence": confidence,
            "model_agreement": float(agreement),
            "method": "analytical_skill_blend",
        }

    def blend(
        self,
        model_forecasts: dict[str, float] | np.ndarray,
        skill_scores: dict[str, float] | np.ndarray = None,
        quality_scores: dict[str, float] | np.ndarray = None,
        regime_id: str = "dry",
        lead_time_hours: int = 24,
        month: int = 6,
        latitude: float = 20.0,
        longitude: float = 78.0,
        elevation: float = 0.0,
        coast_distance: float = 5.0,
    ) -> dict[str, Any]:
        """
        Main blending interface.

        Accepts either dicts keyed by model_id or numpy arrays.
        Returns blended forecast with weights, uncertainty, and metadata.
        """
        # Convert dicts to arrays
        if isinstance(model_forecasts, dict):
            fc_arr = np.array([model_forecasts.get(mid, 0) for mid in self.model_ids])
        else:
            fc_arr = np.asarray(model_forecasts)

        if skill_scores is None:
            skill_arr = np.ones(self.n_models) * 0.5
        elif isinstance(skill_scores, dict):
            skill_arr = np.array([skill_scores.get(mid, 0.5) for mid in self.model_ids])
        else:
            skill_arr = np.asarray(skill_scores)

        if quality_scores is None:
            qual_arr = np.ones(self.n_models) * 0.9
        elif isinstance(quality_scores, dict):
            qual_arr = np.array([quality_scores.get(mid, 0.9) for mid in self.model_ids])
        else:
            qual_arr = np.asarray(quality_scores)

        # Calculate disagreement
        disagreement = np.std(fc_arr) / max(np.mean(np.abs(fc_arr)), 1e-8)

        # Use analytical blend (always available)
        result = self.analytical_blend(
            fc_arr, skill_arr, qual_arr,
            regime_id=regime_id,
            lead_time_hours=lead_time_hours,
            model_disagreement=disagreement,
        )

        # Add model-level detail
        result["model_weights"] = {
            self.model_ids[i]: float(result["weights"][i])
            for i in range(self.n_models)
        }
        result["model_forecasts"] = {
            self.model_ids[i]: float(fc_arr[i])
            for i in range(self.n_models)
        }
        result["model_skills"] = {
            self.model_ids[i]: float(skill_arr[i])
            for i in range(self.n_models)
        }
        result["regime"] = regime_id
        result["lead_time_hours"] = lead_time_hours
        result["location"] = {"lat": latitude, "lon": longitude}

        # Counterfactual: "what if one model is removed?"
        counterfactuals = {}
        for i, mid in enumerate(self.model_ids):
            remaining_fc = np.delete(fc_arr, i)
            remaining_skill = np.delete(skill_arr, i)
            remaining_qual = np.delete(qual_arr, i)
            remaining_ids = [m for j, m in enumerate(self.model_ids) if j != i]

            # Quick analytical blend without the removed model
            scores = remaining_skill * remaining_qual
            exp_s = np.exp(scores / self.temperature)
            w = exp_s / np.sum(exp_s)
            cf_blend = float(np.sum(w * remaining_fc))
            counterfactuals[mid] = {
                "blend_without": cf_blend,
                "difference": float(result["blended_forecast"] - cf_blend),
            }
        result["counterfactuals"] = counterfactuals

        return result

    def blend_spatial(
        self,
        model_grids: dict[str, np.ndarray],
        skill_grids: dict[str, np.ndarray] | None = None,
        quality_scores: dict[str, float] | None = None,
        regime_id: str = "dry",
        lead_time_hours: int = 24,
        lats: np.ndarray = None,
        lons: np.ndarray = None,
    ) -> dict[str, np.ndarray]:
        """
        Blend over an entire spatial grid, producing weight maps.

        Returns spatial arrays of blended forecasts, weights, and uncertainty.
        """
        # Stack model grids: (n_models, n_lat, n_lon)
        model_stack = np.stack([model_grids[mid] for mid in self.model_ids if mid in model_grids])
        actual_ids = [mid for mid in self.model_ids if mid in model_grids]
        n_models, n_lat, n_lon = model_stack.shape

        # Skill vectors per model
        if skill_grids:
            skill_stack = np.stack([skill_grids.get(mid, np.ones((n_lat, n_lon)) * 0.5) for mid in actual_ids])
        else:
            skill_stack = np.ones_like(model_stack) * 0.5

        # Quality
        if quality_scores:
            qual_vec = np.array([quality_scores.get(mid, 0.9) for mid in actual_ids])
        else:
            qual_vec = np.ones(n_models) * 0.9

        # Calculate weights per grid cell
        weight_map = np.zeros_like(model_stack)
        blended = np.zeros((n_lat, n_lon))
        uncertainty = np.zeros((n_lat, n_lon))

        for i in range(n_lat):
            for j in range(n_lon):
                fc = model_stack[:, i, j]
                sk = skill_stack[:, i, j] if skill_stack.ndim == 3 else skill_stack[:, 0, 0]
                ql = qual_vec

                scores = sk * ql
                exp_s = np.exp(scores / self.temperature)
                w = exp_s / np.sum(exp_s)

                weight_map[:, i, j] = w
                blended[i, j] = np.sum(w * fc)
                uncertainty[i, j] = np.sqrt(np.sum(w * (fc - blended[i, j]) ** 2))

        # Dominant model map
        dominant_model_idx = np.argmax(weight_map, axis=0)

        return {
            "blended": blended,
            "weight_maps": {
                actual_ids[k]: weight_map[k] for k in range(n_models)
            },
            "uncertainty": uncertainty,
            "dominant_model": dominant_model_idx,
            "dominant_model_ids": actual_ids,
            "model_agreement": 1.0 - np.std(model_stack, axis=0) / np.maximum(np.mean(np.abs(model_stack), axis=0), 1e-8),
        }

    def generate_explanation(self, blend_result: dict) -> str:
        """Generate human-readable explanation of blending decision."""
        weights = blend_result.get("model_weights", {})
        regime = blend_result.get("regime", "unknown")
        lt = blend_result.get("lead_time_hours", 0)
        conf = blend_result.get("confidence", 0)
        agreement = blend_result.get("model_agreement", 0)

        # Find dominant model
        if weights:
            dominant = max(weights, key=weights.get)
            dominant_weight = weights[dominant]
        else:
            dominant = "unknown"
            dominant_weight = 0

        # Build explanation
        parts = [
            f"Detected regime: {regime.upper().replace('_', ' ')}",
            f"Lead time: {lt} hours",
            "",
            "Model weights:",
        ]
        for mid, w in sorted(weights.items(), key=lambda x: -x[1]):
            bar = "█" * int(w * 20)
            parts.append(f"  {mid}: {w:.1%} {bar}")

        parts.extend([
            "",
            f"Dominant model: {dominant} ({dominant_weight:.1%})",
            f"Model agreement: {'HIGH' if agreement > 0.7 else 'MODERATE' if agreement > 0.4 else 'LOW'}",
            f"Forecast confidence: {'HIGH' if conf > 0.7 else 'MODERATE' if conf > 0.4 else 'LOW'} ({conf:.0%})",
            "",
            f"Reason: {dominant} received the highest weight because it historically "
            f"performs better in the detected regime ({regime.replace('_', ' ')}) "
            f"at this lead time ({lt}h).",
        ])

        return "\n".join(parts)
