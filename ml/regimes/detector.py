"""
SAMYUKTA — Weather Regime Detector

Automatically classifies forecast situations into weather regimes using
unsupervised clustering (PCA + K-Means/HDBSCAN) on meteorological features.
No manual labels required for clustering; semi-supervised refinement optional.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from dataclasses import dataclass, field
from typing import Any
import logging
import json

logger = logging.getLogger(__name__)


# Predefined regime labels for interpretation
REGIME_LABELS = {
    0: "active_monsoon",
    1: "break_monsoon",
    2: "heavy_rainfall",
    3: "convective",
    4: "dry",
    5: "heat_wave",
    6: "high_wind",
    7: "cyclonic",
    8: "western_disturbance",
    9: "coastal_rainfall",
    10: "orographic_rainfall",
}

REGIME_DISPLAY = {
    "active_monsoon": "Active Monsoon",
    "break_monsoon": "Break Monsoon",
    "heavy_rainfall": "Heavy Rainfall",
    "convective": "Convective",
    "dry": "Dry",
    "heat_wave": "Heat Wave",
    "high_wind": "High Wind",
    "cyclonic": "Cyclone / Synoptic Disturbance",
    "western_disturbance": "Western Disturbance",
    "coastal_rainfall": "Coastal Rainfall",
    "orographic_rainfall": "Orographic Rainfall",
}


@dataclass
class RegimeResult:
    """Result of regime detection for a single time step."""
    regime_id: str
    regime_name: str
    confidence: float
    features: dict[str, float]
    cluster_distances: dict[str, float] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "regime_id": self.regime_id,
            "regime_name": self.regime_name,
            "confidence": round(self.confidence, 4),
            "features": {k: round(v, 4) for k, v in self.features.items()},
            "cluster_distances": {k: round(v, 4) for k, v in self.cluster_distances.items()},
            "timestamp": self.timestamp,
        }


class WeatherRegimeDetector:
    """
    Detects weather regimes using unsupervised clustering.

    Pipeline:
    1. Extract meteorological features from forecast/observation fields
    2. Standardize features
    3. PCA for dimensionality reduction
    4. K-Means clustering for regime assignment
    5. Map clusters to interpretable regime labels using feature signatures
    """

    def __init__(
        self,
        n_regimes: int = 8,
        n_pca_components: int = 5,
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.n_pca_components = n_pca_components
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca_components, random_state=random_state)
        self.clusterer = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=10)
        self.is_fitted = False
        self.cluster_to_regime: dict[int, str] = {}
        self.cluster_centers_original: np.ndarray | None = None

    def extract_features(self, ds: xr.Dataset, time_idx: int | None = None) -> dict[str, float]:
        """
        Extract meteorological features from a gridded dataset.

        Features capture the 'state' of the atmosphere to determine regime.
        """
        features = {}

        if time_idx is not None:
            data = {var: ds[var].isel(time=time_idx).values for var in ds.data_vars}
        else:
            data = {var: ds[var].values for var in ds.data_vars}
            # If 3D, take spatial mean
            data = {k: v.mean(axis=0) if v.ndim == 3 else v for k, v in data.items()}

        # Rainfall features
        if "rainfall" in data:
            rain = data["rainfall"]
            valid_rain = rain[~np.isnan(rain)]
            if len(valid_rain) > 0:
                features["rain_mean"] = float(np.mean(valid_rain))
                features["rain_max"] = float(np.max(valid_rain))
                features["rain_std"] = float(np.std(valid_rain))
                features["rain_coverage"] = float(np.mean(valid_rain > 2.5))  # % area with rain
                features["rain_heavy_coverage"] = float(np.mean(valid_rain > 64.5))
                features["rain_p90"] = float(np.percentile(valid_rain, 90))
                features["rain_skewness"] = float(
                    np.mean(((valid_rain - valid_rain.mean()) / max(valid_rain.std(), 1e-8)) ** 3)
                )

        # Temperature features
        if "temperature" in data:
            temp = data["temperature"]
            valid_temp = temp[~np.isnan(temp)]
            if len(valid_temp) > 0:
                features["temp_mean"] = float(np.mean(valid_temp))
                features["temp_max"] = float(np.max(valid_temp))
                features["temp_range"] = float(np.max(valid_temp) - np.min(valid_temp))
                features["temp_std"] = float(np.std(valid_temp))
                features["temp_hot_frac"] = float(np.mean(valid_temp > 40))

        # Wind features
        if "wind_speed" in data:
            wind = data["wind_speed"]
            valid_wind = wind[~np.isnan(wind)]
            if len(valid_wind) > 0:
                features["wind_mean"] = float(np.mean(valid_wind))
                features["wind_max"] = float(np.max(valid_wind))
                features["wind_std"] = float(np.std(valid_wind))
                features["wind_strong_frac"] = float(np.mean(valid_wind > 17.2))

        # Derived features
        if "rain_mean" in features and "temp_mean" in features:
            features["rain_temp_ratio"] = features["rain_mean"] / max(features["temp_mean"], 1)

        if "rain_mean" in features and "wind_mean" in features:
            features["rain_wind_product"] = features["rain_mean"] * features["wind_mean"]

        return features

    def _features_to_matrix(self, feature_list: list[dict]) -> tuple[np.ndarray, list[str]]:
        """Convert list of feature dicts to a consistent numpy array."""
        # Use union of all feature keys
        all_keys = sorted(set(k for f in feature_list for k in f.keys()))
        matrix = np.zeros((len(feature_list), len(all_keys)))
        for i, feat in enumerate(feature_list):
            for j, key in enumerate(all_keys):
                matrix[i, j] = feat.get(key, 0.0)
        return matrix, all_keys

    def fit(self, ds: xr.Dataset) -> "WeatherRegimeDetector":
        """
        Fit regime detector on historical data.

        Extracts features for each timestep, clusters them, then maps
        clusters to interpretable regime labels based on feature signatures.
        """
        n_times = ds.dims.get("time", 1)
        feature_list = []

        for t in range(n_times):
            features = self.extract_features(ds, time_idx=t)
            if features:
                feature_list.append(features)

        if len(feature_list) < self.n_regimes:
            logger.warning(f"Only {len(feature_list)} samples, reducing n_regimes")
            self.n_regimes = max(2, len(feature_list) // 2)
            self.clusterer = KMeans(
                n_clusters=self.n_regimes,
                random_state=self.random_state,
                n_init=10,
            )

        X, self.feature_names = self._features_to_matrix(feature_list)

        # Standardize
        X_scaled = self.scaler.fit_transform(X)

        # PCA
        n_comp = min(self.n_pca_components, X_scaled.shape[1], X_scaled.shape[0])
        self.pca = PCA(n_components=n_comp, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X_scaled)

        # Cluster
        labels = self.clusterer.fit_predict(X_pca)

        # Compute silhouette score
        if len(set(labels)) > 1:
            sil_score = silhouette_score(X_pca, labels)
            logger.info(f"Regime clustering silhouette score: {sil_score:.3f}")

        # Map clusters to regime labels based on feature signatures
        self.cluster_centers_original = self.scaler.inverse_transform(
            self.pca.inverse_transform(self.clusterer.cluster_centers_)
        )
        self._assign_regime_labels(X, labels)

        self.is_fitted = True
        logger.info(f"Regime detector fitted with {self.n_regimes} regimes on {len(feature_list)} samples")
        return self

    def _assign_regime_labels(self, X: np.ndarray, labels: np.ndarray) -> None:
        """
        Map cluster IDs to interpretable regime labels using feature signatures.
        Uses heuristics based on domain knowledge.
        """
        feature_idx = {name: i for i, name in enumerate(self.feature_names)}

        cluster_means = {}
        for c in range(self.n_regimes):
            mask = labels == c
            if mask.sum() > 0:
                cluster_means[c] = X[mask].mean(axis=0)

        # Sort clusters by dominant meteorological signature
        assigned = set()
        self.cluster_to_regime = {}

        for c, means in sorted(cluster_means.items()):
            rain_mean = means[feature_idx.get("rain_mean", 0)] if "rain_mean" in feature_idx else 0
            rain_heavy = means[feature_idx.get("rain_heavy_coverage", 0)] if "rain_heavy_coverage" in feature_idx else 0
            temp_mean = means[feature_idx.get("temp_mean", 0)] if "temp_mean" in feature_idx else 0
            temp_hot = means[feature_idx.get("temp_hot_frac", 0)] if "temp_hot_frac" in feature_idx else 0
            wind_mean = means[feature_idx.get("wind_mean", 0)] if "wind_mean" in feature_idx else 0
            wind_strong = means[feature_idx.get("wind_strong_frac", 0)] if "wind_strong_frac" in feature_idx else 0

            # Heuristic assignment
            if rain_heavy > 0.1 and "heavy_rainfall" not in assigned:
                label = "heavy_rainfall"
            elif wind_strong > 0.15 and "high_wind" not in assigned:
                label = "high_wind"
            elif temp_hot > 0.2 and "heat_wave" not in assigned:
                label = "heat_wave"
            elif rain_mean > 15 and "active_monsoon" not in assigned:
                label = "active_monsoon"
            elif rain_mean > 5 and "convective" not in assigned:
                label = "convective"
            elif rain_mean < 2 and temp_mean > 30 and "dry" not in assigned:
                label = "dry"
            elif rain_mean < 5 and "break_monsoon" not in assigned:
                label = "break_monsoon"
            elif wind_mean > 10 and rain_mean > 10 and "cyclonic" not in assigned:
                label = "cyclonic"
            else:
                # Assign remaining
                remaining = set(REGIME_LABELS.values()) - assigned
                label = remaining.pop() if remaining else f"regime_{c}"

            self.cluster_to_regime[c] = label
            assigned.add(label)

    def predict(self, ds: xr.Dataset, time_idx: int | None = None) -> RegimeResult:
        """Classify a single time step into a weather regime."""
        if not self.is_fitted:
            raise RuntimeError("Regime detector not fitted. Call fit() first.")

        features = self.extract_features(ds, time_idx)
        X, _ = self._features_to_matrix([features])

        # Pad/align features
        aligned = np.zeros((1, len(self.feature_names)))
        for j, name in enumerate(self.feature_names):
            if name in features:
                aligned[0, j] = features[name]

        X_scaled = self.scaler.transform(aligned)
        X_pca = self.pca.transform(X_scaled)

        # Predict cluster
        cluster = self.clusterer.predict(X_pca)[0]
        distances = self.clusterer.transform(X_pca)[0]

        # Confidence based on distance to assigned cluster vs others
        min_dist = distances[cluster]
        sorted_dists = np.sort(distances)
        if len(sorted_dists) > 1 and sorted_dists[1] > 0:
            confidence = 1.0 - (min_dist / sorted_dists[1])
        else:
            confidence = 0.5
        confidence = float(np.clip(confidence, 0.1, 0.99))

        regime_id = self.cluster_to_regime.get(cluster, f"regime_{cluster}")
        regime_name = REGIME_DISPLAY.get(regime_id, regime_id.replace("_", " ").title())

        # Cluster distances for interpretability
        cluster_dists = {
            self.cluster_to_regime.get(i, f"regime_{i}"): float(distances[i])
            for i in range(len(distances))
        }

        timestamp = ""
        if time_idx is not None and "time" in ds.coords:
            timestamp = str(ds.time.values[time_idx])

        return RegimeResult(
            regime_id=regime_id,
            regime_name=regime_name,
            confidence=confidence,
            features=features,
            cluster_distances=cluster_dists,
            timestamp=timestamp,
        )

    def predict_batch(self, ds: xr.Dataset) -> list[RegimeResult]:
        """Classify all timesteps."""
        n_times = ds.dims.get("time", 1)
        return [self.predict(ds, time_idx=t) for t in range(n_times)]

    def get_regime_stats(self, ds: xr.Dataset) -> dict:
        """Compute regime frequency statistics."""
        results = self.predict_batch(ds)
        from collections import Counter
        counts = Counter(r.regime_id for r in results)
        total = len(results)
        return {
            regime: {
                "count": count,
                "frequency": count / total,
                "display_name": REGIME_DISPLAY.get(regime, regime),
            }
            for regime, count in counts.most_common()
        }

    def save(self, path: str) -> None:
        """Save fitted detector parameters."""
        import pickle
        state = {
            "scaler": self.scaler,
            "pca": self.pca,
            "clusterer": self.clusterer,
            "n_regimes": self.n_regimes,
            "feature_names": self.feature_names,
            "cluster_to_regime": self.cluster_to_regime,
            "is_fitted": self.is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "WeatherRegimeDetector":
        """Load a fitted detector."""
        import pickle
        with open(path, "rb") as f:
            state = pickle.load(f)
        det = cls(n_regimes=state["n_regimes"])
        det.scaler = state["scaler"]
        det.pca = state["pca"]
        det.clusterer = state["clusterer"]
        det.feature_names = state["feature_names"]
        det.cluster_to_regime = state["cluster_to_regime"]
        det.is_fitted = state["is_fitted"]
        return det
