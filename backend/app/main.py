"""
SAMYUKTA — FastAPI Backend

REST API for the Hybrid Weather Forecast Fusion system.
All endpoints serve data from the ML pipeline; no data is fabricated.
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from datetime import datetime
from ml.inference.pipeline import SamyuktaPipeline
from backend.app.database import init_db, get_session, DBForecastRun

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Global pipeline instance
pipeline: SamyuktaPipeline | None = None


def persist_run_to_db(run) -> None:
    """Save ForecastRun to database for audit and provenance."""
    try:
        session = get_session()
        ts = datetime.utcnow()
        if hasattr(run, "timestamp") and run.timestamp:
            try:
                ts = datetime.fromisoformat(run.timestamp)
            except Exception:
                ts = datetime.utcnow()

        db_run = DBForecastRun(
            run_id=run.run_id,
            timestamp=ts,
            init_time=ts,
            lead_time_hours=run.lead_time_hours,
            variable=run.variable,
            regime_detected=run.regime_detected,
            regime_confidence=run.regime_confidence,
            blended_forecast_mean=run.blended_forecast_mean,
            uncertainty=run.uncertainty,
            confidence=run.confidence,
            model_agreement=run.model_agreement,
            data_mode=run.data_mode,
            processing_time_ms=run.processing_time_ms,
            explanation=run.explanation,
            models_used=run.models_used,
            models_unavailable=run.models_unavailable,
            model_weights=run.model_weights,
            quality_scores=run.quality_scores,
            skill_scores=run.skill_scores,
            hazard_events=run.hazard_events,
            verification=run.verification,
            counterfactuals=run.counterfactuals,
        )
        session.merge(db_run)
        session.commit()
        session.close()
    except Exception as e:
        logger.warning(f"Database persist error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline and database on startup."""
    global pipeline
    data_dir = os.environ.get("DATA_DIR", str(ROOT / "data" / "demo"))
    logger.info(f"Initializing SAMYUKTA pipeline from {data_dir}")

    # Initialize database tables
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database init warning: {e}")

    pipeline = SamyuktaPipeline(data_dir=data_dir)

    # Check if demo data exists, generate if not
    demo_dir = Path(data_dir)
    if not (demo_dir / "ground_truth.nc").exists():
        logger.info("Demo data not found, generating...")
        from ml.ingestion.synthetic_generator import generate_demo_dataset
        generate_demo_dataset(
            output_dir=data_dir,
            n_days=90,
            start_date="2024-06-01",
            resolution=0.5,
        )

    status = pipeline.initialize()
    logger.info(f"Pipeline init status: {status}")
    yield
    logger.info("SAMYUKTA shutdown")


app = FastAPI(
    title="SAMYUKTA API",
    description="Intelligent Hybrid Weather Forecast Fusion — REST API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Schemas ────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    initialized: bool
    data_mode: str
    models_available: int
    total_runs: int

class ForecastRequest(BaseModel):
    time_idx: int = Field(0, ge=0, description="Time index in the dataset")
    lead_time: int = Field(24, description="Lead time in hours")
    variable: str = Field("rainfall", description="Variable: rainfall, temperature, wind_speed")

class BlendRequest(BaseModel):
    time_idx: int = 0
    lead_time: int = 24
    variable: str = "rainfall"
    exclude_models: list[str] = Field(default_factory=list, description="Models to exclude (for counterfactual)")


# ─── API Routes ───────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """System health check."""
    if pipeline is None:
        return HealthResponse(status="starting", initialized=False, data_mode="UNKNOWN", models_available=0, total_runs=0)
    s = pipeline.get_system_status()
    avail = sum(1 for m in s["models"].values() if m["available"])
    return HealthResponse(
        status="healthy" if s["initialized"] else "degraded",
        initialized=s["initialized"],
        data_mode=s["data_mode"],
        models_available=avail,
        total_runs=s["total_runs"],
    )


@app.get("/models", tags=["Models"])
async def list_models():
    """List all forecast models and their status."""
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")
    status = pipeline.get_system_status()
    return {"models": status["models"], "data_mode": status["data_mode"]}


@app.get("/status", tags=["System"])
async def system_status():
    """Full system status."""
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")
    return pipeline.get_system_status()


@app.post("/forecast/run", tags=["Forecast"])
async def run_forecast(req: ForecastRequest):
    """Execute a single forecast blend run."""
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")
    try:
        run = pipeline.run_forecast(
            time_idx=req.time_idx,
            lead_time=req.lead_time,
            variable=req.variable,
        )
        return run.to_dict()
    except Exception as e:
        logger.error(f"Forecast run error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.get("/forecasts", tags=["Forecast"])
async def list_forecasts(limit: int = Query(20, ge=1, le=100)):
    """List recent forecast runs."""
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")
    runs = pipeline.forecast_runs[-limit:]
    return {"runs": [r.to_dict() for r in reversed(runs)], "total": len(pipeline.forecast_runs)}


@app.get("/forecast/{run_id}", tags=["Forecast"])
async def get_forecast(run_id: str):
    """Get a specific forecast run by ID."""
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")
    for run in pipeline.forecast_runs:
        if run.run_id == run_id:
            return run.to_dict()
    raise HTTPException(404, f"Run {run_id} not found")


@app.get("/weights", tags=["Weights"])
async def get_weights(
    time_idx: int = Query(0, ge=0),
    lead_time: int = Query(24),
    variable: str = Query("rainfall"),
):
    """Get model weight map for visualization."""
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")
    try:
        data = pipeline.get_weight_map_data(time_idx, lead_time, variable)
        if not data:
            raise HTTPException(404, "No weight data available")
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/weight-map", tags=["Weights"])
async def get_weight_map(
    time_idx: int = Query(0, ge=0),
    lead_time: int = Query(24),
    variable: str = Query("rainfall"),
):
    """Get spatial weight map data (same as /weights, alternate endpoint)."""
    return await get_weights(time_idx, lead_time, variable)


@app.get("/regimes", tags=["Regimes"])
async def get_regimes():
    """Get regime statistics and current detected regime."""
    if pipeline is None or not pipeline.is_initialized:
        raise HTTPException(503, "Pipeline not initialized")
    stats = pipeline.regime_detector.get_regime_stats(pipeline.ground_truth)

    # Current regime (latest time)
    n_times = pipeline.ground_truth.dims.get("time", 1)
    current = pipeline.regime_detector.predict(pipeline.ground_truth, time_idx=min(n_times - 1, 0))
    return {
        "current_regime": current.to_dict(),
        "regime_statistics": stats,
        "data_mode": "DEMO / SYNTHETIC DATA",
    }


@app.get("/hazards", tags=["Extreme Weather"])
async def get_hazards(
    time_idx: int = Query(0, ge=0),
    lead_time: int = Query(24),
):
    """Detect extreme weather hazards."""
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")
    try:
        run = pipeline.run_forecast(time_idx=time_idx, lead_time=lead_time, variable="rainfall")
        return {
            "hazard_events": run.hazard_events,
            "regime": run.regime_detected,
            "confidence": run.confidence,
            "data_mode": "DEMO / SYNTHETIC DATA",
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/verification", tags=["Verification"])
async def get_verification(variable: str = Query("rainfall")):
    """Get verification results comparing all methods."""
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")

    # Run a quick verification if none exists
    if not pipeline.verification.results:
        pipeline.run_forecast(time_idx=0, lead_time=24, variable=variable)

    improvement = pipeline.verification.get_improvement_table(variable)
    results = [r.to_dict() for r in pipeline.verification.results if r.variable == variable]
    return {
        "improvement_table": improvement,
        "detailed_results": results,
        "data_mode": "DEMO / SYNTHETIC DATA",
    }


@app.get("/metrics", tags=["Verification"])
async def get_metrics():
    """Get all stored skill metrics."""
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")
    return {"metrics": pipeline.skill_engine.get_all_skills()}


@app.get("/data-health", tags=["Data Health"])
async def get_data_health():
    """Get data source health status."""
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")

    health = {}
    for mid in pipeline.model_ids:
        fc_data = pipeline.model_forecasts.get(mid, {})
        has_data = len(fc_data) > 0
        lead_times = list(fc_data.keys())

        quality = 0.0
        if has_data:
            # Assess quality of first available
            for lt, ds in fc_data.items():
                qr = pipeline.quality_engine.assess(ds, mid)
                quality = qr.quality_score
                break

        health[mid] = {
            "available": has_data,
            "lead_times": lead_times,
            "quality_score": round(quality, 4),
            "status": "ONLINE" if has_data else "OFFLINE",
            "data_mode": "DEMO / SYNTHETIC DATA",
        }

    return {
        "sources": health,
        "ground_truth_available": pipeline.ground_truth is not None,
        "data_mode": "DEMO / SYNTHETIC DATA",
    }


@app.post("/demo/run", tags=["Demo"])
async def run_demo():
    """Execute complete SIH demo scenario."""
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")
    try:
        result = pipeline.run_demo_scenario()
        return result
    except Exception as e:
        logger.error(f"Demo error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.get("/audit/{run_id}", tags=["Audit"])
async def get_audit(run_id: str):
    """Get full audit trail for a forecast run."""
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")
    for run in pipeline.forecast_runs:
        if run.run_id == run_id:
            return {
                "run": run.to_dict(),
                "provenance": {
                    "forecast_timestamp": run.timestamp,
                    "source_models": run.models_used,
                    "model_weights": run.model_weights,
                    "detected_regime": run.regime_detected,
                    "calibration_method": "bias_correction_linear",
                    "uncertainty": run.uncertainty,
                    "quality_scores": run.quality_scores,
                    "data_mode": run.data_mode,
                },
            }
    raise HTTPException(404, f"Run {run_id} not found")


@app.post("/blend", tags=["Forecast"])
async def blend(req: BlendRequest):
    """
    Run blend with optional model exclusion (for model removal experiment).
    """
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")

    # Temporarily modify available models
    original_ids = pipeline.model_ids.copy()
    if req.exclude_models:
        pipeline.blending_engine.model_ids = [m for m in original_ids if m not in req.exclude_models]
        pipeline.blending_engine.n_models = len(pipeline.blending_engine.model_ids)

    try:
        run = pipeline.run_forecast(
            time_idx=req.time_idx,
            lead_time=req.lead_time,
            variable=req.variable,
        )
        result = run.to_dict()
        result["excluded_models"] = req.exclude_models
        return result
    finally:
        # Restore
        pipeline.blending_engine.model_ids = original_ids
        pipeline.blending_engine.n_models = len(original_ids)


@app.post("/evaluate", tags=["Verification"])
async def evaluate(variable: str = Query("rainfall"), lead_time: int = Query(24)):
    """Run evaluation across multiple time steps."""
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")

    n_times = min(pipeline.ground_truth.dims.get("time", 0), 30)
    for t in range(0, n_times, 3):
        pipeline.run_forecast(time_idx=t, lead_time=lead_time, variable=variable)

    improvement = pipeline.verification.get_improvement_table(variable)
    return {"improvement_table": improvement, "runs_evaluated": n_times // 3}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
