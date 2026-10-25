"""
SAMYUKTA — Intelligent Hybrid Weather Forecast Fusion
Core Configuration Loader
"""
from __future__ import annotations

import os
import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = os.environ.get("CONFIG_PATH", str(ROOT_DIR / "configs" / "settings.yaml"))


def load_yaml_config(path: str | None = None) -> dict[str, Any]:
    """Load YAML configuration file."""
    p = Path(path or CONFIG_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class GridConfig:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    resolution: float
    target_crs: str = "EPSG:4326"

    @property
    def lats(self) -> np.ndarray:
        return np.arange(self.lat_min, self.lat_max + self.resolution / 2, self.resolution)

    @property
    def lons(self) -> np.ndarray:
        return np.arange(self.lon_min, self.lon_max + self.resolution / 2, self.resolution)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.lats), len(self.lons))


@dataclass
class ModelConfig:
    id: str
    name: str
    description: str
    type: str
    provider: str
    resolution: float


@dataclass
class VariableConfig:
    id: str
    name: str
    unit: str
    long_name: str
    min_value: float
    max_value: float


@dataclass
class Settings:
    """Centralized application settings parsed from YAML."""
    raw: dict[str, Any]
    grid: GridConfig
    models: list[ModelConfig]
    variables: list[VariableConfig]
    lead_times: list[int]
    seasons: list[dict]
    regions: list[dict]
    weather_regimes: list[dict]
    extreme_thresholds: dict
    alert_thresholds: dict
    quality: dict
    training: dict
    database: dict
    api: dict
    demo: dict

    @classmethod
    def from_yaml(cls, path: str | None = None) -> "Settings":
        raw = load_yaml_config(path)
        g = raw["grid"]
        grid = GridConfig(
            lat_min=g["lat_min"], lat_max=g["lat_max"],
            lon_min=g["lon_min"], lon_max=g["lon_max"],
            resolution=g["resolution"],
            target_crs=g.get("target_crs", "EPSG:4326"),
        )
        models = [ModelConfig(**m) for m in raw["models"]]
        variables = [VariableConfig(**v) for v in raw["variables"]]
        lead_times = raw["lead_times"]["hours"]
        return cls(
            raw=raw,
            grid=grid,
            models=models,
            variables=variables,
            lead_times=lead_times,
            seasons=raw["seasons"],
            regions=raw["regions"],
            weather_regimes=raw["weather_regimes"],
            extreme_thresholds=raw["extreme_thresholds"],
            alert_thresholds=raw["alert_thresholds"],
            quality=raw["quality"],
            training=raw["training"],
            database=raw["database"],
            api=raw["api"],
            demo=raw["demo"],
        )

    def get_model(self, model_id: str) -> ModelConfig | None:
        return next((m for m in self.models if m.id == model_id), None)

    def get_variable(self, var_id: str) -> VariableConfig | None:
        return next((v for v in self.variables if v.id == var_id), None)

    def get_season(self, month: int) -> dict:
        for s in self.seasons:
            if month in s["months"]:
                return s
        return self.seasons[0]

    def get_region_by_point(self, lat: float, lon: float) -> list[dict]:
        """Return all regions containing the given point."""
        matches = []
        for r in self.regions:
            bb = r["bbox"]  # [lat_min, lon_min, lat_max, lon_max]
            if bb[0] <= lat <= bb[2] and bb[1] <= lon <= bb[3]:
                matches.append(r)
        return matches


# Singleton
_settings: Settings | None = None


def get_settings(path: str | None = None) -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_yaml(path)
    return _settings
