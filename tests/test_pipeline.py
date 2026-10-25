"""
SAMYUKTA — Automated Test Suite

Tests for all core components of the forecast blending pipeline.
"""
import sys
import os
import pytest
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ─── Synthetic Generator Tests ───────────────────

class TestSyntheticGenerator:
    def test_ground_truth_generation(self):
        from ml.ingestion.synthetic_generator import SyntheticGroundTruth
        lats = np.arange(6, 38, 2.0)
        lons = np.arange(68, 98, 2.0)
        dates = pd.date_range("2024-06-01", periods=10, freq="D")

        gt = SyntheticGroundTruth(lats, lons, seed=42)
        ds = gt.generate(dates)

        assert "rainfall" in ds.data_vars
        assert "temperature" in ds.data_vars
        assert "wind_speed" in ds.data_vars
        assert ds.dims["time"] == 10
        assert ds.dims["lat"] == len(lats)
        assert ds.dims["lon"] == len(lons)

    def test_rainfall_non_negative(self):
        from ml.ingestion.synthetic_generator import SyntheticGroundTruth
        lats = np.arange(6, 38, 2.0)
        lons = np.arange(68, 98, 2.0)
        dates = pd.date_range("2024-06-01", periods=30, freq="D")
        gt = SyntheticGroundTruth(lats, lons)
        ds = gt.generate(dates)
        assert np.all(ds["rainfall"].values >= 0), "Rainfall must be non-negative"

    def test_model_forecaster(self):
        from ml.ingestion.synthetic_generator import SyntheticGroundTruth, SyntheticModelForecaster
        lats = np.arange(6, 38, 4.0)
        lons = np.arange(68, 98, 4.0)
        dates = pd.date_range("2024-06-01", periods=5, freq="D")

        gt = SyntheticGroundTruth(lats, lons)
        truth = gt.generate(dates)

        for model_id in ["ncum_g", "ncum_r", "gfs", "wrf", "ai_model"]:
            forecaster = SyntheticModelForecaster(model_id, lats, lons)
            fc = forecaster.generate_forecast(truth, lead_time_hours=24)
            assert fc.dims == truth.dims
            assert "rainfall" in fc.data_vars
            # Forecasts should differ from truth (bias/noise)
            assert not np.allclose(fc["rainfall"].values, truth["rainfall"].values)

    def test_different_models_differ(self):
        from ml.ingestion.synthetic_generator import SyntheticGroundTruth, SyntheticModelForecaster
        lats = np.arange(6, 38, 4.0)
        lons = np.arange(68, 98, 4.0)
        dates = pd.date_range("2024-06-01", periods=5, freq="D")

        gt = SyntheticGroundTruth(lats, lons)
        truth = gt.generate(dates)

        fc1 = SyntheticModelForecaster("ncum_r", lats, lons).generate_forecast(truth, 24)
        fc2 = SyntheticModelForecaster("gfs", lats, lons).generate_forecast(truth, 24)

        # Different models should produce different forecasts
        assert not np.allclose(fc1["rainfall"].values, fc2["rainfall"].values)


# ─── Data Quality Tests ───────────────────────

class TestDataQuality:
    def test_quality_assessment(self):
        from ml.preprocessing.pipeline import DataQualityEngine
        engine = DataQualityEngine()

        ds = xr.Dataset({
            "rainfall": (["time", "lat", "lon"], np.random.rand(5, 10, 10) * 50),
        }, coords={
            "time": pd.date_range("2024-06-01", periods=5),
            "lat": np.arange(10, 20, 1.0),
            "lon": np.arange(70, 80, 1.0),
        })

        report = engine.assess(ds, "test_model")
        assert 0 <= report.quality_score <= 1
        assert report.coverage > 0

    def test_nan_detection(self):
        from ml.preprocessing.pipeline import DataQualityEngine
        engine = DataQualityEngine()

        arr = np.random.rand(5, 10, 10) * 50
        arr[0, :5, :5] = np.nan  # inject NaNs

        ds = xr.Dataset({
            "rainfall": (["time", "lat", "lon"], arr),
        }, coords={
            "time": pd.date_range("2024-06-01", periods=5),
            "lat": np.arange(10, 20, 1.0),
            "lon": np.arange(70, 80, 1.0),
        })

        report = engine.assess(ds, "test_model")
        assert report.has_nans


# ─── Regime Detector Tests ─────────────────────

class TestRegimeDetector:
    def test_fit_and_predict(self):
        from ml.regimes.detector import WeatherRegimeDetector
        lats = np.arange(6, 38, 4.0)
        lons = np.arange(68, 98, 4.0)
        dates = pd.date_range("2024-01-01", periods=60, freq="D")

        # Create synthetic data with varying patterns
        n_lat, n_lon = len(lats), len(lons)
        rainfall = np.random.gamma(2, 10, size=(60, n_lat, n_lon)).astype(np.float32)
        temperature = np.random.normal(30, 5, size=(60, n_lat, n_lon)).astype(np.float32)
        wind_speed = np.random.exponential(5, size=(60, n_lat, n_lon)).astype(np.float32)

        ds = xr.Dataset({
            "rainfall": (["time", "lat", "lon"], rainfall),
            "temperature": (["time", "lat", "lon"], temperature),
            "wind_speed": (["time", "lat", "lon"], wind_speed),
        }, coords={"time": dates, "lat": lats, "lon": lons})

        detector = WeatherRegimeDetector(n_regimes=4)
        detector.fit(ds)
        assert detector.is_fitted

        result = detector.predict(ds, time_idx=0)
        assert result.regime_id is not None
        assert 0 < result.confidence <= 1

    def test_regime_labels(self):
        from ml.regimes.detector import REGIME_LABELS, REGIME_DISPLAY
        for k, v in REGIME_LABELS.items():
            assert v in REGIME_DISPLAY


# ─── Skill Engine Tests ──────────────────────

class TestSkillEngine:
    def test_metrics_calculation(self):
        from ml.skill.engine import HistoricalSkillEngine
        engine = HistoricalSkillEngine()

        fc = np.random.rand(100) * 50
        obs = fc + np.random.randn(100) * 5  # small error

        metrics = engine.calculate_metrics(fc, obs, "rainfall")
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "correlation" in metrics
        assert 0 <= metrics["skill_score"] <= 1

    def test_skill_storage(self):
        from ml.skill.engine import HistoricalSkillEngine
        engine = HistoricalSkillEngine()

        ds_fc = xr.Dataset({
            "rainfall": (["time", "lat"], np.random.rand(10, 5) * 50),
        })
        ds_obs = xr.Dataset({
            "rainfall": (["time", "lat"], np.random.rand(10, 5) * 50),
        })

        skill = engine.update_skill("ncum_r", ds_fc, ds_obs, "rainfall", lead_time=24)
        assert skill.model_id == "ncum_r"
        assert skill.variable == "rainfall"

        retrieved = engine.get_skill("ncum_r", "rainfall", lead_time=24)
        assert retrieved is not None


# ─── Blending Engine Tests ────────────────────

class TestBlendingEngine:
    def test_weight_normalization(self):
        from ml.blending.rsc_moe import AdaptiveBlendingEngine
        engine = AdaptiveBlendingEngine(model_ids=["a", "b", "c"])

        result = engine.blend(
            model_forecasts={"a": 10.0, "b": 20.0, "c": 15.0},
            skill_scores={"a": 0.8, "b": 0.6, "c": 0.7},
        )

        weights = result["weights"]
        assert np.isclose(np.sum(weights), 1.0), "Weights must sum to 1"
        assert np.all(weights >= 0), "Weights must be non-negative"

    def test_blend_result_structure(self):
        from ml.blending.rsc_moe import AdaptiveBlendingEngine
        engine = AdaptiveBlendingEngine(model_ids=["ncum_g", "ncum_r", "gfs"])

        result = engine.blend(
            model_forecasts={"ncum_g": 50.0, "ncum_r": 60.0, "gfs": 45.0},
        )

        assert "blended_forecast" in result
        assert "confidence" in result
        assert "model_weights" in result
        assert "model_agreement" in result
        assert "counterfactuals" in result

    def test_counterfactuals(self):
        from ml.blending.rsc_moe import AdaptiveBlendingEngine
        engine = AdaptiveBlendingEngine(model_ids=["a", "b", "c"])

        result = engine.blend(
            model_forecasts={"a": 10.0, "b": 20.0, "c": 15.0},
        )

        assert "a" in result["counterfactuals"]
        assert "blend_without" in result["counterfactuals"]["a"]

    def test_spatial_blend(self):
        from ml.blending.rsc_moe import AdaptiveBlendingEngine
        engine = AdaptiveBlendingEngine(model_ids=["a", "b"])

        grids = {
            "a": np.random.rand(10, 10) * 50,
            "b": np.random.rand(10, 10) * 50,
        }

        result = engine.blend_spatial(grids)
        assert "blended" in result
        assert result["blended"].shape == (10, 10)
        assert "weight_maps" in result


# ─── Extreme Weather Tests ────────────────────

class TestExtremeWeather:
    def test_heavy_rainfall_detection(self):
        from ml.extremes.engine import ExtremeWeatherEngine
        engine = ExtremeWeatherEngine()

        lats = np.arange(10, 20, 1.0)
        lons = np.arange(70, 80, 1.0)

        # Create data with heavy rainfall
        blended = np.ones((10, 10)) * 100  # 100mm everywhere
        models = {"a": np.ones((10, 10)) * 95, "b": np.ones((10, 10)) * 105}

        events = engine.detect_heavy_rainfall(blended, models, lats, lons)
        assert len(events) > 0
        assert events[0].hazard_type == "heavy_rainfall"
        assert 0 <= events[0].probability <= 1

    def test_no_false_extreme(self):
        from ml.extremes.engine import ExtremeWeatherEngine
        engine = ExtremeWeatherEngine()

        lats = np.arange(10, 20, 1.0)
        lons = np.arange(70, 80, 1.0)

        # Low rainfall — should not trigger
        blended = np.ones((10, 10)) * 5  # 5mm
        models = {"a": np.ones((10, 10)) * 4, "b": np.ones((10, 10)) * 6}

        events = engine.detect_heavy_rainfall(blended, models, lats, lons)
        assert len(events) == 0


# ─── Physics Validation Tests ──────────────────

class TestPhysicsValidation:
    def test_rainfall_clipping(self):
        from ml.preprocessing.pipeline import PhysicsValidator
        ds = xr.Dataset({"rainfall": (["x"], np.array([-5, 0, 10, 500, 1500]))})
        ds = PhysicsValidator.validate(ds)
        assert ds["rainfall"].values.min() >= 0
        assert ds["rainfall"].values.max() <= 1000

    def test_weight_normalization(self):
        from ml.preprocessing.pipeline import PhysicsValidator
        weights = np.array([[0.3, 0.5, -0.1], [0.2, 0.8, 0.3]])
        normalized = PhysicsValidator.validate_weights(weights)
        assert np.all(normalized >= 0)
        # Each column should sum to 1
        assert np.allclose(normalized.sum(axis=0), 1.0)


# ─── Verification Tests ──────────────────────

class TestVerification:
    def test_evaluation(self):
        from ml.evaluation.verification import VerificationFramework
        vf = VerificationFramework()

        fc_a = np.random.rand(100) * 50
        fc_b = np.random.rand(100) * 50
        blended = (fc_a + fc_b) / 2
        obs = fc_a * 0.6 + fc_b * 0.4 + np.random.randn(100) * 3

        results = vf.evaluate(
            model_forecasts={"model_a": fc_a, "model_b": fc_b},
            blended_forecast=blended,
            observation=obs,
            variable="rainfall",
        )

        assert len(results) >= 3  # 2 models + equal ensemble + SAMYUKTA
        samyukta = [r for r in results if r.method_name == "SAMYUKTA"]
        assert len(samyukta) == 1

    def test_improvement_table(self):
        from ml.evaluation.verification import VerificationFramework
        vf = VerificationFramework()

        fc = {"a": np.random.rand(50) * 30, "b": np.random.rand(50) * 30}
        obs = np.random.rand(50) * 30
        blended = np.mean(list(fc.values()), axis=0)

        vf.evaluate(fc, blended, obs, "rainfall")
        table = vf.get_improvement_table("rainfall")
        assert len(table) > 0


# ─── Bias Correction Tests ────────────────────

class TestBiasCorrection:
    def test_linear_correction(self):
        from ml.preprocessing.pipeline import BiasCorrector
        bc = BiasCorrector()

        # Create biased forecast
        fc = xr.Dataset({
            "temperature": (["time", "lat"], np.random.rand(20, 5) * 10 + 35),
        })
        obs = xr.Dataset({
            "temperature": (["time", "lat"], np.random.rand(20, 5) * 10 + 30),
        })

        bc.fit("test_model", fc, obs)
        corrected = bc.correct("test_model", fc, method="linear")

        # Corrected should be closer to obs range
        assert abs(corrected["temperature"].values.mean() - obs["temperature"].values.mean()) < \
               abs(fc["temperature"].values.mean() - obs["temperature"].values.mean())


# ─── Configuration Tests ──────────────────────

class TestConfig:
    def test_load_config(self):
        from backend.app.core.config import Settings
        settings = Settings.from_yaml(str(ROOT / "configs" / "settings.yaml"))
        assert settings.grid.lat_min == 6.0
        assert len(settings.models) == 5
        assert len(settings.variables) == 3

    def test_season_lookup(self):
        from backend.app.core.config import Settings
        settings = Settings.from_yaml(str(ROOT / "configs" / "settings.yaml"))
        season = settings.get_season(7)  # July
        assert season["id"] == "sw_monsoon"


# ─── Uncertainty Engine Tests ──────────────────

class TestUncertaintyEngine:
    def test_ensemble_spread_uncertainty(self):
        from ml.uncertainty.engine import UncertaintyEngine
        engine = UncertaintyEngine(confidence_level=0.90)

        forecasts = np.array([25.0, 30.0, 28.0, 32.0, 27.0])
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        est = engine.from_ensemble_spread(forecasts, weights=weights, variable="temperature")
        assert est.deterministic == pytest.approx(28.4, abs=0.1)
        assert est.lower_bound < est.deterministic < est.upper_bound
        assert 0.0 <= est.uncertainty_score <= 1.0
        assert est.method == "ensemble_spread"

    def test_rainfall_non_negative_lower_bound(self):
        from ml.uncertainty.engine import UncertaintyEngine
        engine = UncertaintyEngine(confidence_level=0.95)

        forecasts = np.array([0.5, 1.0, 0.2, 2.0])
        est = engine.from_ensemble_spread(forecasts, variable="rainfall")
        assert est.lower_bound >= 0.0, "Rainfall uncertainty lower bound must be >= 0"

    def test_conformal_calibration_and_prediction(self):
        from ml.uncertainty.engine import UncertaintyEngine
        engine = UncertaintyEngine(confidence_level=0.90)

        residuals = np.random.normal(0, 3.0, size=100)
        engine.calibrate_conformal(residuals, variable="temperature")

        est = engine.from_conformal(forecast=30.0, variable="temperature")
        assert est.lower_bound < 30.0 < est.upper_bound
        assert est.method == "conformal_prediction"

    def test_spatial_uncertainty(self):
        from ml.uncertainty.engine import UncertaintyEngine
        engine = UncertaintyEngine(confidence_level=0.90)

        grid_a = np.ones((5, 5)) * 20.0
        grid_b = np.ones((5, 5)) * 30.0
        grids = {"model_a": grid_a, "model_b": grid_b}

        spatial = engine.spatial_uncertainty(grids, variable="temperature")
        assert spatial["deterministic"].shape == (5, 5)
        assert np.allclose(spatial["deterministic"], 25.0)
        assert np.all(spatial["lower_bound"] < spatial["upper_bound"])


# ─── Calibration Module Tests ──────────────────

class TestCalibrationModule:
    def test_probability_calibrator_platt(self):
        from ml.calibration.calibrator import ProbabilityCalibrator
        cal = ProbabilityCalibrator(method="platt")

        raw_probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        outcomes = np.array([0, 0, 0, 1, 1, 1])
        cal.fit(raw_probs, outcomes)
        assert cal.is_fitted

        cal_probs = cal.calibrate(np.array([0.15, 0.85]))
        assert len(cal_probs) == 2
        assert 0.0 <= cal_probs[0] <= 1.0
        assert 0.0 <= cal_probs[1] <= 1.0
        assert cal_probs[0] < cal_probs[1]

    def test_quantile_mapping_calibrator(self):
        from ml.calibration.calibrator import QuantileMappingCalibrator
        qm = QuantileMappingCalibrator(n_quantiles=50)

        fc = np.linspace(10, 40, 100)
        obs = np.linspace(8, 32, 100)
        qm.fit(fc, obs)
        assert qm.is_fitted

        calibrated = qm.calibrate(np.array([10.0, 25.0, 40.0]))
        assert np.isclose(calibrated[0], 8.0, atol=1.0)
        assert np.isclose(calibrated[2], 32.0, atol=1.0)

    def test_spread_calibrator(self):
        from ml.calibration.calibrator import SpreadCalibrator
        sc = SpreadCalibrator()

        spreads = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        errors = np.array([4.0, 4.0, 4.0, 4.0, 4.0])
        sc.fit(spreads, errors)
        assert sc.is_fitted
        assert sc.scale_factor > 1.0
        scaled = sc.calibrate_spread(2.0)
        assert scaled > 2.0

    def test_calibration_engine_pipeline(self):
        from ml.calibration.calibrator import CalibrationEngine
        ce = CalibrationEngine()

        fc = np.random.uniform(5, 50, size=100)
        obs = fc * 0.9 + 2.0
        metrics = ce.fit_variable("rainfall", fc, obs, threshold=25.0)

        assert metrics.variable == "rainfall"
        assert metrics.rmse_after <= metrics.rmse_before + 0.1
        cal_fc = ce.calibrate_forecast(fc[:5], "rainfall")
        assert len(cal_fc) == 5


# ─── Backend API Tests ────────────────────────

class TestBackendAPI:
    def test_api_health_endpoint(self):
        from fastapi.testclient import TestClient
        from backend.app.main import app

        client = TestClient(app)
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
        assert "data_mode" in data

    def test_api_models_endpoint_error_handling(self):
        from fastapi.testclient import TestClient
        from backend.app.main import app
        import backend.app.main as main_mod

        client = TestClient(app)
        # When pipeline is uninitialized
        orig = main_mod.pipeline
        main_mod.pipeline = None
        try:
            res = client.get("/models")
            assert res.status_code == 503
        finally:
            main_mod.pipeline = orig


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

