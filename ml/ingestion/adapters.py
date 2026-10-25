"""
SAMYUKTA — Model Adapter Architecture

Abstract base classes and concrete adapters for weather forecast sources.
Every adapter exposes a common interface regardless of the underlying data source.
"""
from __future__ import annotations

import abc
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)


class ModelAdapter(abc.ABC):
    """
    Base adapter interface for all weather forecast sources.

    Every adapter must implement:
    - get_forecast(): returns an xr.Dataset of forecasts
    - get_metadata(): returns model metadata dict
    - check_availability(): returns True if source is available
    """

    def __init__(self, model_id: str, model_name: str, model_type: str):
        self.model_id = model_id
        self.model_name = model_name
        self.model_type = model_type  # 'nwp', 'ai', 'file', 'synthetic'
        self._available = True

    @abc.abstractmethod
    def get_forecast(
        self,
        init_time: datetime,
        lead_times: list[int],
        variables: list[str],
        lat_range: tuple[float, float] | None = None,
        lon_range: tuple[float, float] | None = None,
    ) -> xr.Dataset:
        """Retrieve forecast data."""
        ...

    @abc.abstractmethod
    def check_availability(self) -> bool:
        """Check if the data source is currently available."""
        ...

    def get_metadata(self) -> dict[str, Any]:
        """Return model metadata."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "available": self._available,
        }


class NWPModelAdapter(ModelAdapter):
    """
    Adapter for Numerical Weather Prediction models (NCUM-G, NCUM-R, GFS, WRF).
    In production, this would connect to NCMRWF/NCEP data feeds.
    Currently implements file-based ingestion of precomputed outputs.
    """

    def __init__(self, model_id: str, model_name: str, data_dir: str | Path):
        super().__init__(model_id, model_name, model_type="nwp")
        self.data_dir = Path(data_dir)

    def get_forecast(
        self,
        init_time: datetime,
        lead_times: list[int],
        variables: list[str],
        lat_range: tuple[float, float] | None = None,
        lon_range: tuple[float, float] | None = None,
    ) -> xr.Dataset:
        """Load NWP forecast from file."""
        datasets = []
        for lt in lead_times:
            fname = f"forecast_{self.model_id}_lt{lt:03d}.nc"
            fpath = self.data_dir / fname
            if fpath.exists():
                ds = xr.open_dataset(fpath)
                if lat_range:
                    ds = ds.sel(lat=slice(*lat_range))
                if lon_range:
                    ds = ds.sel(lon=slice(*lon_range))
                # Select variables
                avail_vars = [v for v in variables if v in ds.data_vars]
                if avail_vars:
                    ds = ds[avail_vars]
                    ds.attrs["lead_time_hours"] = lt
                    ds.attrs["init_time"] = init_time.isoformat()
                    datasets.append(ds)
            else:
                logger.warning(f"Forecast file not found: {fpath}")

        if not datasets:
            self._available = False
            raise FileNotFoundError(f"No forecast files found for {self.model_id}")

        self._available = True
        return datasets

    def check_availability(self) -> bool:
        self._available = self.data_dir.exists() and any(self.data_dir.glob(f"forecast_{self.model_id}_*.nc"))
        return self._available


class AIWeatherModelAdapter(ModelAdapter):
    """
    Adapter for AI-based weather prediction models (e.g., GraphCast, Pangu-Weather).
    In production, this would run inference or call an API.
    Currently uses precomputed outputs.
    """

    def __init__(self, model_id: str, model_name: str, data_dir: str | Path):
        super().__init__(model_id, model_name, model_type="ai")
        self.data_dir = Path(data_dir)

    def get_forecast(
        self,
        init_time: datetime,
        lead_times: list[int],
        variables: list[str],
        lat_range: tuple[float, float] | None = None,
        lon_range: tuple[float, float] | None = None,
    ) -> xr.Dataset:
        return NWPModelAdapter.get_forecast(self, init_time, lead_times, variables, lat_range, lon_range)

    def check_availability(self) -> bool:
        self._available = self.data_dir.exists() and any(self.data_dir.glob(f"forecast_{self.model_id}_*.nc"))
        return self._available


class SyntheticForecastAdapter(ModelAdapter):
    """
    Adapter that generates synthetic forecasts on-the-fly for demo mode.
    Clearly labels all output as DEMO / SYNTHETIC DATA.
    """

    def __init__(self, model_id: str, model_name: str, lats: np.ndarray, lons: np.ndarray):
        super().__init__(model_id, model_name, model_type="synthetic")
        self.lats = lats
        self.lons = lons
        # Lazy import to avoid circular
        from ml.ingestion.synthetic_generator import SyntheticModelForecaster, SyntheticGroundTruth
        self._gt_gen = SyntheticGroundTruth(lats, lons)
        self._forecaster = SyntheticModelForecaster(model_id, lats, lons)

    def get_forecast(
        self,
        init_time: datetime,
        lead_times: list[int],
        variables: list[str],
        lat_range: tuple[float, float] | None = None,
        lon_range: tuple[float, float] | None = None,
    ) -> list[xr.Dataset]:
        dates = pd.DatetimeIndex([init_time])
        truth = self._gt_gen.generate(dates, variables)
        datasets = []
        for lt in lead_times:
            fc = self._forecaster.generate_forecast(truth, lt, variables)
            fc.attrs["init_time"] = init_time.isoformat()
            fc.attrs["lead_time_hours"] = lt
            fc.attrs["data_mode"] = "DEMO / SYNTHETIC DATA"
            datasets.append(fc)
        return datasets

    def check_availability(self) -> bool:
        self._available = True
        return True


class ObservationAdapter:
    """
    Adapter for observation/reference data (station obs, satellite, reanalysis).
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def get_observations(
        self,
        date_range: tuple[datetime, datetime],
        variables: list[str],
    ) -> xr.Dataset:
        gt_path = self.data_dir / "ground_truth.nc"
        if gt_path.exists():
            ds = xr.open_dataset(gt_path)
            ds = ds.sel(time=slice(date_range[0], date_range[1]))
            avail = [v for v in variables if v in ds.data_vars]
            return ds[avail] if avail else ds
        raise FileNotFoundError(f"Observation data not found at {gt_path}")


class ModelAdapterRegistry:
    """Registry to manage all model adapters."""

    def __init__(self):
        self._adapters: dict[str, ModelAdapter] = {}

    def register(self, adapter: ModelAdapter) -> None:
        self._adapters[adapter.model_id] = adapter
        logger.info(f"Registered adapter: {adapter.model_id} ({adapter.model_type})")

    def get(self, model_id: str) -> ModelAdapter | None:
        return self._adapters.get(model_id)

    def list_available(self) -> list[str]:
        return [mid for mid, a in self._adapters.items() if a.check_availability()]

    def list_all(self) -> list[dict]:
        return [a.get_metadata() for a in self._adapters.values()]

    def get_all_forecasts(
        self,
        init_time: datetime,
        lead_times: list[int],
        variables: list[str],
    ) -> dict[str, list[xr.Dataset]]:
        """Get forecasts from all available models."""
        results = {}
        for model_id, adapter in self._adapters.items():
            try:
                if adapter.check_availability():
                    results[model_id] = adapter.get_forecast(
                        init_time, lead_times, variables
                    )
                else:
                    logger.warning(f"Model {model_id} unavailable, skipping")
            except Exception as e:
                logger.error(f"Error getting forecast from {model_id}: {e}")
        return results
