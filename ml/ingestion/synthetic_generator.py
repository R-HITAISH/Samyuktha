"""
SAMYUKTA — Synthetic Demo Data Generator

Generates physically plausible multi-model weather forecasts over the Indian domain.
Each model has intentionally different bias structures, noise levels, and regime-
dependent skill so the adaptive blender can demonstrate its value.

All generated data is clearly labeled as DEMO / SYNTHETIC DATA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import json
import logging

logger = logging.getLogger(__name__)


class SyntheticGroundTruth:
    """
    Generates a spatially and temporally coherent 'ground truth' over India.
    This is NOT real observation data — it is a plausible synthetic reference.
    """

    def __init__(self, lats: np.ndarray, lons: np.ndarray, seed: int = 42):
        self.lats = lats
        self.lons = lons
        self.rng = np.random.default_rng(seed)
        self.lat_grid, self.lon_grid = np.meshgrid(lats, lons, indexing="ij")

    def _elevation_proxy(self) -> np.ndarray:
        """Synthetic elevation: higher in north (Himalayas), Western Ghats ridge."""
        elev = np.zeros_like(self.lat_grid)
        # Himalayan belt
        elev += 3000 * np.exp(-((self.lat_grid - 32) ** 2) / 20)
        # Western Ghats
        mask_wg = (self.lon_grid >= 73) & (self.lon_grid <= 76) & \
                  (self.lat_grid >= 8) & (self.lat_grid <= 20)
        elev[mask_wg] += 800
        return elev

    def _coast_distance_proxy(self) -> np.ndarray:
        """Rough proxy: distance from edges of Indian domain."""
        d_west = self.lon_grid - 68
        d_east = 98 - self.lon_grid
        d_south = self.lat_grid - 6
        d_north = 38 - self.lat_grid
        return np.minimum(np.minimum(d_west, d_east), np.minimum(d_south, d_north))

    def generate(
        self,
        dates: pd.DatetimeIndex,
        variables: list[str] = None,
    ) -> xr.Dataset:
        """Generate ground truth dataset for all dates and variables."""
        if variables is None:
            variables = ["rainfall", "temperature", "wind_speed"]

        n_lat, n_lon = len(self.lats), len(self.lons)
        n_time = len(dates)
        elev = self._elevation_proxy()
        coast_d = self._coast_distance_proxy()

        data_vars = {}
        for var in variables:
            arr = np.zeros((n_time, n_lat, n_lon), dtype=np.float32)
            for t, date in enumerate(dates):
                month = date.month
                doy = date.day_of_year
                # Seasonal cycle
                season_phase = np.sin(2 * np.pi * (doy - 172) / 365)  # peak ~June 21

                if var == "rainfall":
                    # Base monsoon pattern: peaks in SW monsoon, spatial variation
                    monsoon_factor = np.clip(season_phase, 0, 1)
                    # Latitude dependence: more rain in tropics
                    lat_factor = np.exp(-((self.lat_grid - 20) ** 2) / 200)
                    # Western Ghats enhancement
                    wg_factor = np.where(
                        (self.lon_grid >= 73) & (self.lon_grid <= 76) &
                        (self.lat_grid >= 8) & (self.lat_grid <= 20),
                        2.5, 1.0
                    )
                    # East coast NE monsoon
                    ne_factor = np.where(
                        (month in [10, 11, 12]) &
                        (self.lon_grid >= 78) & (self.lat_grid <= 16),
                        1.5, 1.0
                    )
                    base = 15 * monsoon_factor * lat_factor * wg_factor * ne_factor
                    # Add intermittency and spatial noise
                    noise = self.rng.gamma(2, 3, size=(n_lat, n_lon))
                    rain_event = self.rng.random((n_lat, n_lon)) > (0.4 + 0.3 * monsoon_factor)
                    arr[t] = np.clip(base + noise * rain_event, 0, None)
                    # Add occasional extreme events
                    if self.rng.random() < 0.05 and n_lat > 2 and n_lon > 2:
                        r = min(8, max(1, min(n_lat, n_lon) // 4))
                        min_x, max_x = (r, n_lat - r) if n_lat > 2 * r else (0, n_lat)
                        min_y, max_y = (r, n_lon - r) if n_lon > 2 * r else (0, n_lon)
                        cx = self.rng.integers(min_x, max(min_x + 1, max_x))
                        cy = self.rng.integers(min_y, max(min_y + 1, max_y))
                        yy, xx = np.ogrid[-r:r+1, -r:r+1]
                        mask = xx**2 + yy**2 <= r**2
                        s = arr[t, max(0,cx-r):cx+r+1, max(0,cy-r):cy+r+1]
                        m = mask[:s.shape[0], :s.shape[1]]
                        s[m] += self.rng.uniform(80, 200)

                elif var == "temperature":
                    # Base: warmer in south/plains, cooler in north/mountains
                    lat_base = 35 - 0.5 * (self.lat_grid - 10)
                    elev_cool = -6.5 * elev / 1000  # lapse rate
                    season_temp = 8 * np.cos(2 * np.pi * (doy - 150) / 365)
                    noise = self.rng.normal(0, 1.5, size=(n_lat, n_lon))
                    arr[t] = lat_base + elev_cool + season_temp + noise
                    # Heat wave events
                    if month in [4, 5, 6] and self.rng.random() < 0.08:
                        hw_mask = (self.lat_grid >= 22) & (self.lat_grid <= 32) & \
                                  (self.lon_grid >= 74) & (self.lon_grid <= 86)
                        arr[t][hw_mask] += self.rng.uniform(4, 8)

                elif var == "wind_speed":
                    base = 3 + 2 * np.abs(season_phase)
                    coast_boost = np.clip(5 - coast_d, 0, 5)
                    noise = self.rng.exponential(1.5, size=(n_lat, n_lon))
                    arr[t] = base + coast_boost + noise
                    # Cyclone-like wind event
                    if month in [10, 11, 5, 6] and self.rng.random() < 0.03 and n_lat > 2 and n_lon > 2:
                        r = min(6, max(1, min(n_lat, n_lon) // 4))
                        min_x, max_x = (r, n_lat - r) if n_lat > 2 * r else (0, n_lat)
                        min_y, max_y = (r, n_lon - r) if n_lon > 2 * r else (0, n_lon)
                        cx = self.rng.integers(min_x, max(min_x + 1, max_x))
                        cy = self.rng.integers(min_y, max(min_y + 1, max_y))
                        yy, xx = np.ogrid[-r:r+1, -r:r+1]
                        mask = xx**2 + yy**2 <= r**2
                        s = arr[t, max(0,cx-r):cx+r+1, max(0,cy-r):cy+r+1]
                        m = mask[:s.shape[0], :s.shape[1]]
                        s[m] += self.rng.uniform(15, 35)

            data_vars[var] = (["time", "lat", "lon"], arr)

        ds = xr.Dataset(
            data_vars,
            coords={
                "time": dates,
                "lat": self.lats,
                "lon": self.lons,
            },
            attrs={
                "source": "SAMYUKTA Synthetic Ground Truth",
                "data_mode": "DEMO / SYNTHETIC DATA",
                "created": datetime.utcnow().isoformat(),
                "description": "Physically plausible synthetic reference. NOT real observations.",
            },
        )
        return ds


class SyntheticModelForecaster:
    """
    Creates synthetic model forecasts by perturbing ground truth with model-specific
    biases, noise, lead-time degradation, and regime-dependent errors.

    Each model has unique characteristics so the blending system can learn preferences.
    """

    # Model-specific bias profiles
    MODEL_PROFILES = {
        "ncum_g": {
            "rainfall_bias": 1.1,    # slight wet bias
            "temp_bias": -0.3,       # slight cool bias
            "wind_bias": 0.5,        # slight high bias
            "noise_scale": 1.0,
            "lead_time_decay": 0.04, # skill degrades with lead time
            "monsoon_skill": 0.85,   # good in monsoon
            "dry_skill": 0.90,       # good in dry
            "extreme_sensitivity": 0.7,
            "spatial_bias_pattern": "uniform",
        },
        "ncum_r": {
            "rainfall_bias": 0.95,
            "temp_bias": 0.1,
            "wind_bias": -0.2,
            "noise_scale": 0.8,      # lower noise = higher skill generally
            "lead_time_decay": 0.06, # faster skill decay (regional model)
            "monsoon_skill": 0.92,   # best in monsoon
            "dry_skill": 0.80,
            "extreme_sensitivity": 0.85,  # good at extremes
            "spatial_bias_pattern": "coastal_good",
        },
        "gfs": {
            "rainfall_bias": 0.85,   # dry bias
            "temp_bias": 0.5,        # warm bias
            "wind_bias": 0.3,
            "noise_scale": 1.2,
            "lead_time_decay": 0.03, # slow decay = good at longer lead
            "monsoon_skill": 0.75,
            "dry_skill": 0.88,
            "extreme_sensitivity": 0.6,
            "spatial_bias_pattern": "north_good",
        },
        "wrf": {
            "rainfall_bias": 1.2,    # wet bias
            "temp_bias": -0.5,
            "wind_bias": -0.3,
            "noise_scale": 1.1,
            "lead_time_decay": 0.07, # fast decay
            "monsoon_skill": 0.80,
            "dry_skill": 0.70,
            "extreme_sensitivity": 0.9,  # good at extremes
            "spatial_bias_pattern": "mountain_good",
        },
        "ai_model": {
            "rainfall_bias": 1.0,    # low bias
            "temp_bias": 0.05,
            "wind_bias": 0.1,
            "noise_scale": 0.7,      # lowest noise
            "lead_time_decay": 0.02, # very slow decay
            "monsoon_skill": 0.88,
            "dry_skill": 0.92,
            "extreme_sensitivity": 0.5,  # poor at extremes (common AI issue)
            "spatial_bias_pattern": "uniform",
        },
    }

    def __init__(self, model_id: str, lats: np.ndarray, lons: np.ndarray, seed: int = None):
        self.model_id = model_id
        self.profile = self.MODEL_PROFILES[model_id]
        self.lats = lats
        self.lons = lons
        self.lat_grid, self.lon_grid = np.meshgrid(lats, lons, indexing="ij")
        self.rng = np.random.default_rng(seed or hash(model_id) % (2**31))

    def _spatial_bias_field(self) -> np.ndarray:
        """Generate model-specific spatial bias pattern."""
        pattern = self.profile["spatial_bias_pattern"]
        field = np.ones_like(self.lat_grid)
        if pattern == "coastal_good":
            coast_d = np.minimum(
                np.minimum(self.lon_grid - 68, 98 - self.lon_grid),
                np.minimum(self.lat_grid - 6, 38 - self.lat_grid)
            )
            field = 1.0 - 0.3 * np.exp(-coast_d / 3)  # lower bias near coast
        elif pattern == "north_good":
            field = 1.0 - 0.2 * (self.lat_grid - 6) / 32
        elif pattern == "mountain_good":
            elev_proxy = np.exp(-((self.lat_grid - 32) ** 2) / 20)
            field = 1.0 - 0.2 * elev_proxy
        return field

    def generate_forecast(
        self,
        truth: xr.Dataset,
        lead_time_hours: int,
        variables: list[str] = None,
    ) -> xr.Dataset:
        """
        Generate model forecast by perturbing ground truth.

        Lead time causes:
        - increased noise
        - increased bias
        - reduced extreme capture
        """
        if variables is None:
            variables = list(truth.data_vars)

        lead_factor = 1.0 + self.profile["lead_time_decay"] * lead_time_hours
        spatial_bias = self._spatial_bias_field()

        data_vars = {}
        for var in variables:
            truth_arr = truth[var].values.copy()
            n_time, n_lat, n_lon = truth_arr.shape
            forecast = np.zeros_like(truth_arr)

            for t in range(n_time):
                gt = truth_arr[t]

                if var == "rainfall":
                    # Multiplicative bias + noise
                    bias = self.profile["rainfall_bias"] * spatial_bias
                    noise = self.rng.gamma(
                        2, self.profile["noise_scale"] * lead_factor * 2,
                        size=(n_lat, n_lon)
                    )
                    # Dampen extremes based on model sensitivity
                    extreme_mask = gt > np.percentile(gt[gt > 0], 95) if np.any(gt > 0) else np.zeros_like(gt, dtype=bool)
                    dampened = gt.copy()
                    dampened[extreme_mask] *= self.profile["extreme_sensitivity"]

                    forecast[t] = np.clip(dampened * bias + noise * lead_factor - 3, 0, None)

                elif var == "temperature":
                    bias = self.profile["temp_bias"] * spatial_bias * lead_factor
                    noise = self.rng.normal(
                        0, self.profile["noise_scale"] * lead_factor,
                        size=(n_lat, n_lon)
                    )
                    forecast[t] = gt + bias + noise

                elif var == "wind_speed":
                    bias = self.profile["wind_bias"] * spatial_bias
                    noise = self.rng.exponential(
                        self.profile["noise_scale"] * lead_factor * 0.5,
                        size=(n_lat, n_lon)
                    )
                    forecast[t] = np.clip(gt + bias + noise, 0, None)

            data_vars[var] = (["time", "lat", "lon"], forecast.astype(np.float32))

        ds = xr.Dataset(
            data_vars,
            coords=truth.coords,
            attrs={
                "source": f"SAMYUKTA Synthetic Forecast — {self.model_id}",
                "model_id": self.model_id,
                "lead_time_hours": lead_time_hours,
                "data_mode": "DEMO / SYNTHETIC DATA",
                "created": datetime.utcnow().isoformat(),
            },
        )
        return ds


def generate_demo_dataset(
    output_dir: str | Path,
    n_days: int = 365,
    start_date: str = "2024-01-01",
    resolution: float = 0.25,
    lat_range: tuple = (6.0, 38.0),
    lon_range: tuple = (68.0, 98.0),
    lead_times: list[int] = None,
    model_ids: list[str] = None,
) -> dict[str, Path]:
    """
    Generate complete demo dataset: ground truth + multi-model forecasts.

    Returns dict of file paths written.
    """
    if lead_times is None:
        lead_times = [6, 12, 24, 48, 72, 120]
    if model_ids is None:
        model_ids = list(SyntheticModelForecaster.MODEL_PROFILES.keys())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lats = np.arange(lat_range[0], lat_range[1] + resolution / 2, resolution)
    lons = np.arange(lon_range[0], lon_range[1] + resolution / 2, resolution)
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    logger.info(f"Generating ground truth: {n_days} days, grid {len(lats)}x{len(lons)}")

    # Ground truth
    gt_gen = SyntheticGroundTruth(lats, lons, seed=42)
    truth = gt_gen.generate(dates)

    files = {}

    # Save ground truth
    gt_path = output_dir / "ground_truth.nc"
    truth.to_netcdf(gt_path)
    files["ground_truth"] = gt_path
    logger.info(f"Saved ground truth: {gt_path}")

    # Generate model forecasts for each lead time
    for model_id in model_ids:
        for lt in lead_times:
            logger.info(f"Generating {model_id} forecast, lead_time={lt}h")
            forecaster = SyntheticModelForecaster(
                model_id, lats, lons,
                seed=hash(f"{model_id}_{lt}") % (2**31)
            )
            fc = forecaster.generate_forecast(truth, lt)
            fc_path = output_dir / f"forecast_{model_id}_lt{lt:03d}.nc"
            fc.to_netcdf(fc_path)
            files[f"{model_id}_lt{lt}"] = fc_path

    # Save metadata
    meta = {
        "data_mode": "DEMO / SYNTHETIC DATA",
        "created": datetime.utcnow().isoformat(),
        "n_days": n_days,
        "start_date": start_date,
        "resolution": resolution,
        "lat_range": list(lat_range),
        "lon_range": list(lon_range),
        "lead_times": lead_times,
        "models": model_ids,
        "variables": ["rainfall", "temperature", "wind_speed"],
        "grid_shape": [len(lats), len(lons)],
        "WARNING": "This is synthetic demo data. NOT real observations or forecasts.",
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    files["metadata"] = meta_path

    logger.info(f"Demo dataset generation complete. {len(files)} files written to {output_dir}")
    return files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_demo_dataset(
        output_dir="data/demo",
        n_days=90,   # 3 months for quick demo
        start_date="2024-06-01",  # Monsoon season
        resolution=0.5,  # Coarser for speed
    )
