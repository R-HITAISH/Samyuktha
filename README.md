# SAMYUKTA — Intelligent Hybrid Weather Forecast Fusion

<p align="center">
  <strong>SIH26081 | Ministry of Earth Sciences | NCMRWF</strong><br>
  <em>Regime-Skill-Conditioned Mixture of Experts for Adaptive Multi-Model Forecast Blending</em>
</p>

---

## 🌟 Problem

Different weather models (NCUM-G, NCUM-R, GFS, WRF, AI models) produce different forecasts.
**No single model is universally best** — their accuracy depends on:

- Geographic region (coastal vs. inland vs. mountains)
- Season (monsoon vs. winter)
- Forecast lead time (6h vs. 72h)
- Weather regime (active monsoon vs. heat wave)
- Variable (rainfall vs. temperature)

Current ensemble methods use **static, uniform weights** — they don't adapt.

## 💡 Solution: RSC-MoE

SAMYUKTA implements a **Regime-Skill-Conditioned Mixture of Experts (RSC-MoE)** that dynamically determines:

> *"Which model should I trust, where, when, and under what atmospheric condition?"*

The system produces **spatially varying, temporally adaptive model weights** conditioned on 9 factors:

1. ✅ Region (lat/lon, elevation, coast distance)
2. ✅ Forecast lead time
3. ✅ Month/season
4. ✅ Weather regime (auto-detected)
5. ✅ Weather variable
6. ✅ Recent model skill
7. ✅ Model disagreement
8. ✅ Data quality
9. ✅ Forecast uncertainty

---

## 🏗️ Architecture

```
INGEST → VALIDATE → REGRID → ALIGN → QC → BIAS CORRECT →
REGIME DETECT → SKILL LOOKUP → RSC-MoE META-LEARNER →
DYNAMIC WEIGHTS → BLEND → UNCERTAINTY → EXTREME EVENTS →
VERIFY → API → DASHBOARD
```

### Core Components

| Component | Implementation | Status |
|-----------|---------------|--------|
| Synthetic Demo Generator | `ml/ingestion/synthetic_generator.py` | ✅ IMPLEMENTED |
| Model Adapter Architecture | `ml/ingestion/adapters.py` | ✅ IMPLEMENTED |
| Data Quality Engine | `ml/preprocessing/pipeline.py` | ✅ IMPLEMENTED |
| Spatial Regridding | `ml/preprocessing/pipeline.py` | ✅ IMPLEMENTED |
| Bias Correction | `ml/preprocessing/pipeline.py` | ✅ IMPLEMENTED |
| Weather Regime Detector | `ml/regimes/detector.py` | ✅ IMPLEMENTED |
| Historical Skill Engine | `ml/skill/engine.py` | ✅ IMPLEMENTED |
| RSC-MoE Blending Engine | `ml/blending/rsc_moe.py` | ✅ IMPLEMENTED |
| Uncertainty Quantification | `ml/uncertainty/engine.py` | ✅ IMPLEMENTED |
| Meteorological Calibration | `ml/calibration/calibrator.py` | ✅ IMPLEMENTED |
| Extreme Weather Engine | `ml/extremes/engine.py` | ✅ IMPLEMENTED |
| Verification Framework | `ml/evaluation/verification.py` | ✅ IMPLEMENTED |
| Master Pipeline | `ml/inference/pipeline.py` | ✅ IMPLEMENTED |
| FastAPI Backend | `backend/app/main.py` | ✅ IMPLEMENTED |
| Next.js Dashboard | `frontend/src/app/page.tsx` | ✅ IMPLEMENTED |
| Docker Setup | `docker-compose.yml` | ✅ IMPLEMENTED |
| Automated Tests | `tests/test_pipeline.py` | ✅ IMPLEMENTED |

---

## 🚀 Quick Start

### 1. Generate Demo Data

```bash
pip install -r requirements.txt
python scripts/seed_demo_data.py
```

### 2. Run the Demo

```bash
python scripts/run_demo.py
```

### 3. Start the Backend

```bash
python -m uvicorn backend.app.main:app --reload --port 8000
```

### 4. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

### 5. Open Dashboard

Navigate to `http://localhost:3000`

### Docker (One Command)

```bash
docker compose up --build
```

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health check |
| GET | `/models` | List all forecast models |
| GET | `/status` | Full system status |
| POST | `/forecast/run` | Execute forecast blend |
| GET | `/forecasts` | List recent runs |
| GET | `/forecast/{id}` | Get specific run |
| GET | `/weights` | Weight map data |
| GET | `/weight-map` | Spatial weight maps |
| GET | `/regimes` | Weather regime info |
| GET | `/hazards` | Extreme weather detection |
| GET | `/verification` | Verification results |
| GET | `/metrics` | Skill metrics |
| GET | `/data-health` | Data source health |
| POST | `/demo/run` | Run SIH demo scenario |
| GET | `/audit/{run_id}` | Forecast audit trail |
| POST | `/blend` | Blend with model exclusion |
| POST | `/evaluate` | Run full evaluation |

API docs available at `http://localhost:8000/docs`

---

## 🌦️ Weather Regimes

SAMYUKTA automatically classifies forecast situations using unsupervised clustering:

- Active Monsoon
- Break Monsoon
- Heavy Rainfall
- Convective
- Dry
- Heat Wave
- High Wind
- Cyclone/Synoptic Disturbance
- Western Disturbance
- Coastal Rainfall
- Orographic Rainfall

---

## 📈 Verification

The system compares these methods:

1. **Best Single Model** — reference baseline
2. **Equal-Weight Ensemble** — standard practice
3. **Static Skill-Weighted** — fixed historical weights
4. **★ SAMYUKTA** — adaptive RSC-MoE blend

Metrics: MAE, RMSE, Bias, Correlation, Skill Score, CSI, POD, FAR, Brier Score

---

## ⚠️ Data Mode

The application clearly distinguishes:

| Mode | Label | Description |
|------|-------|-------------|
| LIVE | 🟢 LIVE | Connected to operational data feeds |
| HISTORICAL | 🔵 HISTORICAL | Using archived/reanalysis data |
| DEMO | 🟡 DEMO / SYNTHETIC DATA | Synthetic demo mode |

All demo data is clearly labeled and never presented as operational forecasts.

---

## 🔬 What Makes SAMYUKTA Different

1. **Weather-Regime Awareness** — not just blending numbers; understanding the atmospheric state
2. **Spatially Varying Weights** — model trust varies by location
3. **Lead-Time Intelligence** — weights adapt as forecast horizon extends
4. **Model Disagreement Signal** — high disagreement → reduced confidence
5. **Counterfactual Explanations** — "what if we remove this model?"
6. **Explainable Decisions** — every weight is justified
7. **Measurable Verification** — improvement demonstrated, not claimed
8. **Physics-Aware Constraints** — no negative rainfall, bounded probabilities
9. **Operational Workflow** — complete pipeline from ingestion to dashboard
10. **Audit Trail** — every forecast run is fully inspectable

---

## 📁 Project Structure

```
samyukta/
├── frontend/          # Next.js dashboard
├── backend/           # FastAPI REST API
│   └── app/
│       ├── core/      # Configuration
│       └── main.py    # API server
├── ml/                # Machine learning pipeline
│   ├── ingestion/     # Data adapters & synthetic generator
│   ├── preprocessing/ # Quality, regridding, bias correction
│   ├── regimes/       # Weather regime detector
│   ├── skill/         # Historical skill engine
│   ├── blending/      # RSC-MoE adaptive engine
│   ├── extremes/      # Extreme weather engine
│   ├── evaluation/    # Verification framework
│   └── inference/     # Master pipeline
├── data/demo/         # Synthetic demo data
├── configs/           # Configuration files
├── scripts/           # Utility scripts
├── tests/             # Automated tests
├── docker/            # Dockerfiles
└── docker-compose.yml # One-command deployment
```

---

## 📋 SIH Requirement Traceability

| SIH Requirement | SAMYUKTA Feature | Evidence |
|----------------|------------------|----------|
| Dynamic blended forecast | RSC-MoE adaptive engine | Forecast tab, API `/forecast/run` |
| Model weight maps | Spatial weight engine | Model Weights tab |
| Improved forecast skill | Verification framework | Verification Lab tab |
| Extreme weather guidance | Hazard engine | Extreme Weather Center |
| Operational workflow | Pipeline + API + Dashboard | Command Center + Demo |
| Rainfall forecast | Multi-model rainfall blend | Forecast tab |
| Temperature forecast | Multi-model temperature blend | Forecast tab |
| Wind forecast | Multi-model wind blend | Forecast tab |
| Uncertainty estimation | Prediction intervals | All forecasts |
| Model comparison | Side-by-side verification | Verification Lab |

---

## ⚖️ License

MIT License

---

<p align="center">
  <strong>SAMYUKTA</strong> — <em>Unified Intelligence for Weather Forecasting</em>
</p>
