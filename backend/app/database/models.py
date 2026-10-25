"""
SAMYUKTA — Database Models (SQLAlchemy ORM)

Stores forecast runs, skill scores, weight history, hazard events,
and audit logs for the complete forecast provenance chain.
"""
from __future__ import annotations

import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime,
    Boolean, Text, JSON, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.pool import StaticPool

Base = declarative_base()

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///data/samyukta.db")


class ForecastModel(Base):
    """Registered forecast model/source."""
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    model_type = Column(String(20))  # nwp, ai, synthetic
    provider = Column(String(50))
    resolution = Column(Float)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ForecastRun(Base):
    """Record of a single pipeline execution."""
    __tablename__ = "forecast_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    init_time = Column(DateTime)
    lead_time_hours = Column(Integer)
    variable = Column(String(30))
    regime_detected = Column(String(50))
    regime_confidence = Column(Float)
    blended_forecast_mean = Column(Float)
    uncertainty = Column(Float)
    confidence = Column(Float)
    model_agreement = Column(Float)
    data_mode = Column(String(20), default="DEMO")
    processing_time_ms = Column(Float)
    explanation = Column(Text)
    models_used = Column(JSON)
    models_unavailable = Column(JSON)
    model_weights = Column(JSON)
    quality_scores = Column(JSON)
    skill_scores = Column(JSON)
    hazard_events = Column(JSON)
    verification = Column(JSON)
    counterfactuals = Column(JSON)


class SkillScore(Base):
    """Historical skill scores for model/variable/region/lead-time combinations."""
    __tablename__ = "skill_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(50), nullable=False, index=True)
    variable = Column(String(30), nullable=False)
    region_id = Column(String(50), default="all")
    lead_time = Column(Integer, default=0)
    season = Column(String(30), default="all")
    weather_regime = Column(String(50), default="all")
    sample_count = Column(Integer, default=0)
    mae = Column(Float)
    rmse = Column(Float)
    bias = Column(Float)
    correlation = Column(Float)
    brier_score = Column(Float)
    csi = Column(Float)
    pod = Column(Float)
    far = Column(Float)
    skill_score = Column(Float)
    last_updated = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_skill_lookup', 'model_id', 'variable', 'region_id', 'lead_time', 'season', 'weather_regime'),
    )


class WeatherRegime(Base):
    """Detected weather regime records."""
    __tablename__ = "weather_regimes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, index=True)
    regime_id = Column(String(50), nullable=False)
    regime_name = Column(String(100))
    confidence = Column(Float)
    features = Column(JSON)
    run_id = Column(String(50))


class HazardEvent(Base):
    """Detected hazard events."""
    __tablename__ = "hazard_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    hazard_type = Column(String(30), nullable=False)
    severity = Column(String(20))
    probability = Column(Float)
    confidence = Column(Float)
    affected_area_fraction = Column(Float)
    peak_value = Column(Float)
    threshold_used = Column(Float)
    alert_level = Column(String(20))
    region = Column(String(50))
    run_id = Column(String(50))
    details = Column(JSON)


class DataSource(Base):
    """Data source health records."""
    __tablename__ = "data_sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String(50), nullable=False, index=True)
    source_type = Column(String(30))
    status = Column(String(20))
    quality_score = Column(Float)
    last_updated = Column(DateTime)
    lead_times_available = Column(JSON)
    issues = Column(JSON)


class DataQuality(Base):
    """Data quality assessment records."""
    __tablename__ = "data_quality"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    coverage = Column(Float)
    missing_fraction = Column(Float)
    quality_score = Column(Float)
    has_nans = Column(Boolean)
    value_range_valid = Column(Boolean)
    stale = Column(Boolean)
    issues = Column(JSON)


class AuditLog(Base):
    """Audit trail for all operations."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    operation = Column(String(50), nullable=False)
    run_id = Column(String(50))
    details = Column(JSON)
    status = Column(String(20))


class VerificationResult(Base):
    """Stored verification results."""
    __tablename__ = "verification_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    method_name = Column(String(50), nullable=False)
    variable = Column(String(30))
    mae = Column(Float)
    rmse = Column(Float)
    bias = Column(Float)
    correlation = Column(Float)
    skill_score = Column(Float)
    lead_time = Column(Integer)
    season = Column(String(30))
    regime = Column(String(50))
    region = Column(String(50))


def init_db(url: str = None):
    """Initialize database and create all tables."""
    db_url = url or DATABASE_URL
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False} if "sqlite" in db_url else {},
        poolclass=StaticPool if "sqlite" in db_url else None,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return engine, Session


def get_session(url: str = None):
    _, Session = init_db(url)
    return Session()
