'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar, LineChart, Line, AreaChart, Area,
  PieChart, Pie, Cell,
} from 'recharts';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ─── Types ────────────────────────────────────
interface ForecastRun {
  run_id: string;
  timestamp: string;
  init_time: string;
  lead_time_hours: number;
  variable: string;
  models_used: string[];
  models_unavailable: string[];
  regime_detected: string;
  regime_confidence: number;
  model_weights: Record<string, number>;
  blended_forecast_mean: number;
  uncertainty: number;
  confidence: number;
  model_agreement: number;
  quality_scores: Record<string, number>;
  skill_scores: Record<string, number>;
  hazard_events: any;
  verification: Record<string, any>;
  explanation: string;
  data_mode: string;
  processing_time_ms: number;
  counterfactuals: Record<string, any>;
}

// ─── API Helper ────────────────────────────────
async function apiFetch(endpoint: string, options: any = {}): Promise<any> {
  try {
    const res = await fetch(`${API}${endpoint}`, {
      ...options,
      headers: { 'Content-Type': 'application/json', ...options.headers },
    });
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json();
  } catch (e) {
    console.error(`API call failed: ${endpoint}`, e);
    return null;
  }
}

// ─── Color Palette ────────────────────────────
const MODEL_COLORS: Record<string, string> = {
  ncum_g: '#3b82f6',
  ncum_r: '#22c55e',
  gfs: '#f59e0b',
  wrf: '#a855f7',
  ai_model: '#06b6d4',
};
const MODEL_NAMES: Record<string, string> = {
  ncum_g: 'NCUM-G',
  ncum_r: 'NCUM-R',
  gfs: 'GFS',
  wrf: 'WRF',
  ai_model: 'AI-Weather',
};

const SEVERITY_COLORS = {
  INFO: '#3b82f6',
  WATCH: '#f59e0b',
  HIGH: '#f97316',
  CRITICAL: '#ef4444',
  NONE: '#6b7280',
};

// ─── Sub-Components ──────────────────────────

function Logo() {
  return (
    <div className="flex items-center gap-3">
      <div className="relative">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center font-black text-white text-lg shadow-lg shadow-blue-500/25">
          S
        </div>
        <div className="absolute -top-0.5 -right-0.5 w-3 h-3 rounded-full bg-green-400 border-2 border-[#0a0e1a] animate-pulse" />
      </div>
      <div>
        <h1 className="text-lg font-bold tracking-tight gradient-text">SAMYUKTA</h1>
        <p className="text-[10px] text-gray-500 font-medium tracking-wider uppercase">Hybrid Weather Intelligence</p>
      </div>
    </div>
  );
}

function DemoBadge() {
  return (
    <span className="demo-badge flex items-center gap-1.5">
      <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
      DEMO / SYNTHETIC DATA
    </span>
  );
}

function KPICard({ title, value, subtitle, color = 'blue', icon }: {
  title: string; value: string | number; subtitle?: string; color?: string; icon?: React.ReactNode;
}) {
  const colorMap: Record<string, string> = {
    blue: 'from-blue-500/10 to-blue-500/5 border-blue-500/20',
    green: 'from-green-500/10 to-green-500/5 border-green-500/20',
    amber: 'from-amber-500/10 to-amber-500/5 border-amber-500/20',
    red: 'from-red-500/10 to-red-500/5 border-red-500/20',
    purple: 'from-purple-500/10 to-purple-500/5 border-purple-500/20',
    cyan: 'from-cyan-500/10 to-cyan-500/5 border-cyan-500/20',
  };
  const textColor: Record<string, string> = {
    blue: 'text-blue-400', green: 'text-green-400', amber: 'text-amber-400',
    red: 'text-red-400', purple: 'text-purple-400', cyan: 'text-cyan-400',
  };
  return (
    <div className={`rounded-2xl bg-gradient-to-br ${colorMap[color]} border p-5 transition-all hover:scale-[1.02]`}>
      <div className="flex justify-between items-start mb-3">
        <p className="text-xs font-medium text-gray-400 uppercase tracking-wider">{title}</p>
        {icon && <div className={`${textColor[color]} opacity-60`}>{icon}</div>}
      </div>
      <p className={`text-2xl font-bold ${textColor[color]}`}>{value}</p>
      {subtitle && <p className="text-xs text-gray-500 mt-1">{subtitle}</p>}
    </div>
  );
}

function WeightBar({ model, weight, skill, quality }: {
  model: string; weight: number; skill: number; quality: number;
}) {
  const pct = Math.round(weight * 100);
  return (
    <div className="group p-3 rounded-xl bg-white/[0.02] hover:bg-white/[0.04] transition-all">
      <div className="flex justify-between items-center mb-2">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: MODEL_COLORS[model] || '#6b7280' }} />
          <span className="text-sm font-semibold">{MODEL_NAMES[model] || model}</span>
        </div>
        <span className="text-sm font-mono font-bold" style={{ color: MODEL_COLORS[model] }}>{pct}%</span>
      </div>
      <div className="weight-bar">
        <div className="weight-bar-fill" style={{ width: `${pct}%`, background: MODEL_COLORS[model] }} />
      </div>
      <div className="flex gap-4 mt-2 text-[11px] text-gray-500">
        <span>Skill: {(skill * 100).toFixed(0)}%</span>
        <span>Quality: {(quality * 100).toFixed(0)}%</span>
      </div>
    </div>
  );
}

function HazardCard({ type, events }: { type: string; events: any[] }) {
  if (!events || events.length === 0) return null;
  const topEvent = events.reduce((a: any, b: any) => (a.probability > b.probability ? a : b), events[0]);
  const typeNames: Record<string, string> = {
    rainfall: '🌧️ Heavy Rainfall',
    heat_wave: '🌡️ Heat Wave',
    high_wind: '💨 High Wind',
  };
  const alertClass = `alert-${(topEvent.alert_level || 'info').toLowerCase()}`;
  return (
    <div className={`glass-card p-4 ${topEvent.alert_level === 'CRITICAL' ? 'glow-red' : ''}`}>
      <div className="flex justify-between items-start mb-3">
        <h4 className="font-semibold text-sm">{typeNames[type] || type}</h4>
        <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${alertClass}`}>
          {topEvent.alert_level || 'INFO'}
        </span>
      </div>
      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-gray-400">Probability</span>
          <span className="font-bold">{(topEvent.probability * 100).toFixed(0)}%</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Severity</span>
          <span className="capitalize">{topEvent.severity}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Peak Value</span>
          <span className="font-mono">{topEvent.peak_value?.toFixed(1)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Area Affected</span>
          <span>{(topEvent.affected_area_fraction * 100).toFixed(1)}%</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">Duration Est.</span>
          <span>{topEvent.duration_estimate}</span>
        </div>
      </div>
      {topEvent.supporting_models?.length > 0 && (
        <div className="mt-3 pt-3 border-t border-white/5">
          <p className="text-[11px] text-gray-500">Supporting Models:</p>
          <div className="flex gap-1 mt-1 flex-wrap">
            {topEvent.supporting_models.map((m: string) => (
              <span key={m} className="px-2 py-0.5 rounded-full text-[10px] bg-white/5 text-gray-300">
                {MODEL_NAMES[m] || m}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Demo Progress ───────────────────────────
function DemoProgress({ steps, currentStep }: { steps: string[]; currentStep: number }) {
  return (
    <div className="space-y-2">
      {steps.map((step, i) => (
        <div key={i} className="flex items-center gap-3">
          <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold
            ${i < currentStep ? 'bg-green-500/20 text-green-400 border border-green-500/30' :
              i === currentStep ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30 animate-pulse' :
              'bg-gray-700/50 text-gray-500 border border-gray-600/30'}`}>
            {i < currentStep ? '✓' : i + 1}
          </div>
          <span className={`text-sm ${i < currentStep ? 'text-green-400' : i === currentStep ? 'text-blue-400' : 'text-gray-500'}`}>
            {step}
          </span>
        </div>
      ))}
    </div>
  );
}

// ─── Main Dashboard ──────────────────────────
const TABS = [
  'Command Center', 'Forecast', 'Model Weights', 'Extreme Weather',
  'Verification', 'Data Health', 'Audit',
];

const DEMO_STEPS = [
  'Load Demo Scenario',
  'Simulate Ingestion',
  'Validate Sources',
  'Detect Weather Regime',
  'Calculate Historical Skill',
  'Generate Adaptive Weights',
  'Produce Blended Forecast',
  'Calculate Uncertainty',
  'Detect Extreme Events',
  'Update Visualization',
  'Generate Explanation',
  'Complete ✓',
];

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState(0);
  const [forecastRun, setForecastRun] = useState<ForecastRun | null>(null);
  const [demoResult, setDemoResult] = useState<any>(null);
  const [demoRunning, setDemoRunning] = useState(false);
  const [demoStep, setDemoStep] = useState(-1);
  const [systemStatus, setSystemStatus] = useState<any>(null);
  const [verification, setVerification] = useState<any>(null);
  const [dataHealth, setDataHealth] = useState<any>(null);
  const [regimes, setRegimes] = useState<any>(null);
  const [hazards, setHazards] = useState<any>(null);
  const [selectedVariable, setSelectedVariable] = useState('rainfall');
  const [selectedLeadTime, setSelectedLeadTime] = useState(24);
  const [selectedTimeIdx, setSelectedTimeIdx] = useState(0);
  const [loading, setLoading] = useState(false);
  const [evaluatorMode, setEvaluatorMode] = useState(false);
  const [excludeModel, setExcludeModel] = useState<string | null>(null);

  // Initial data load
  useEffect(() => {
    loadSystemStatus();
  }, []);

  const loadSystemStatus = async () => {
    const status = await apiFetch('/status');
    if (status) setSystemStatus(status);
  };

  const runForecast = async () => {
    setLoading(true);
    const result = await apiFetch('/forecast/run', {
      method: 'POST',
      body: JSON.stringify({
        time_idx: selectedTimeIdx,
        lead_time: selectedLeadTime,
        variable: selectedVariable,
      }),
    });
    if (result) {
      setForecastRun(result);
      loadVerification();
      loadDataHealth();
      loadRegimes();
    }
    setLoading(false);
  };

  const runDemo = async () => {
    setDemoRunning(true);
    setDemoStep(0);

    // Animate through steps
    for (let i = 0; i < DEMO_STEPS.length - 1; i++) {
      setDemoStep(i);
      await new Promise(r => setTimeout(r, 400));
    }

    const result = await apiFetch('/demo/run', { method: 'POST' });
    if (result) {
      setDemoResult(result);
      // Extract primary run
      if (result.primary_run) {
        setForecastRun(result.primary_run);
      } else if (result.forecast_runs) {
        const firstKey = Object.keys(result.forecast_runs)[0];
        if (firstKey) setForecastRun(result.forecast_runs[firstKey]);
      }
    }
    setDemoStep(DEMO_STEPS.length);
    setDemoRunning(false);

    // Load supporting data
    loadVerification();
    loadDataHealth();
    loadRegimes();
  };

  const loadVerification = async () => {
    const v = await apiFetch(`/verification?variable=${selectedVariable}`);
    if (v) setVerification(v);
  };

  const loadDataHealth = async () => {
    const h = await apiFetch('/data-health');
    if (h) setDataHealth(h);
  };

  const loadRegimes = async () => {
    const r = await apiFetch('/regimes');
    if (r) setRegimes(r);
  };

  const runModelRemoval = async (modelToRemove: string) => {
    setExcludeModel(modelToRemove);
    setLoading(true);
    const result = await apiFetch('/blend', {
      method: 'POST',
      body: JSON.stringify({
        time_idx: selectedTimeIdx,
        lead_time: selectedLeadTime,
        variable: selectedVariable,
        exclude_models: [modelToRemove],
      }),
    });
    setLoading(false);
    if (result) {
      return result;
    }
    return null;
  };

  const runEvaluatorMode = async () => {
    setEvaluatorMode(true);
    setActiveTab(0);
    await runDemo();
    // Cycle through key screens
    const screens = [0, 1, 2, 3, 4, 5];
    for (const screen of screens) {
      setActiveTab(screen);
      await new Promise(r => setTimeout(r, 3000));
    }
    setEvaluatorMode(false);
  };

  // ─── Prepare chart data ─────────────────────
  const weightChartData = forecastRun ? Object.entries(forecastRun.model_weights).map(([model, weight]) => ({
    model: MODEL_NAMES[model] || model,
    weight: Math.round(weight * 100),
    skill: Math.round((forecastRun.skill_scores[model] || 0) * 100),
    quality: Math.round((forecastRun.quality_scores[model] || 0) * 100),
    fill: MODEL_COLORS[model] || '#6b7280',
  })) : [];

  const verificationChartData = verification?.improvement_table?.map((row: any) => ({
    method: row.method === 'SAMYUKTA' ? '★ SAMYUKTA' : row.method.replace('_', '-'),
    RMSE: row.rmse,
    MAE: row.mae,
    'Skill Score': row.skill_score,
    Correlation: row.correlation,
  })) || [];

  const radarData = forecastRun ? Object.entries(forecastRun.model_weights).map(([model, weight]) => ({
    subject: MODEL_NAMES[model] || model,
    Weight: Math.round(weight * 100),
    Skill: Math.round((forecastRun.skill_scores[model] || 0) * 100),
    Quality: Math.round((forecastRun.quality_scores[model] || 0) * 100),
  })) : [];

  // ─── Render ────────────────────────────────
  return (
    <div className="min-h-screen flex flex-col">
      {/* ─── Top Navigation ────────────────── */}
      <header className="sticky top-0 z-50 bg-[#0a0e1a]/90 backdrop-blur-xl border-b border-white/5">
        <div className="max-w-[1800px] mx-auto px-6 py-3 flex items-center justify-between">
          <Logo />

          {/* Navigation */}
          <nav className="hidden lg:flex items-center gap-1">
            {TABS.map((tab, i) => (
              <button
                key={tab}
                onClick={() => setActiveTab(i)}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-all ${
                  activeTab === i
                    ? 'bg-blue-500/10 text-blue-400 nav-active'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
                }`}
              >
                {tab}
              </button>
            ))}
          </nav>

          {/* Right controls */}
          <div className="flex items-center gap-3">
            <DemoBadge />
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              <span>{systemStatus?.initialized ? 'System Online' : 'Initializing'}</span>
            </div>
            <button
              onClick={runDemo}
              disabled={demoRunning}
              className="px-4 py-2 rounded-xl bg-gradient-to-r from-blue-500 to-cyan-500 text-white text-sm font-semibold hover:shadow-lg hover:shadow-blue-500/25 transition-all disabled:opacity-50"
            >
              {demoRunning ? '⏳ Running...' : '▶ RUN SIH DEMO'}
            </button>
          </div>
        </div>
      </header>

      {/* ─── Main Content ──────────────────── */}
      <main className="flex-1 max-w-[1800px] mx-auto w-full px-6 py-6">

        {/* ═══ TAB 0: Command Center ═══ */}
        {activeTab === 0 && (
          <div className="space-y-6">
            {/* Demo Progress */}
            {demoStep >= 0 && (
              <div className="glass-card p-6">
                <h3 className="text-lg font-bold mb-4 gradient-text">SIH Demo Scenario: Heavy Rainfall Event</h3>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  <DemoProgress steps={DEMO_STEPS} currentStep={demoStep} />
                </div>
              </div>
            )}

            {/* KPI Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-4">
              <KPICard title="Forecast Skill" value={forecastRun ? `${(forecastRun.confidence * 100).toFixed(0)}%` : '—'} color="blue" />
              <KPICard
                title="Skill Improvement"
                value={verification?.improvement_table?.find((r: any) => r.method === 'Equal Ensemble')?.skill_improvement_pct
                  ? `+${verification.improvement_table.find((r: any) => r.method === 'Equal Ensemble').skill_improvement_pct}%`
                  : '—'}
                color="green"
              />
              <KPICard
                title="Model Agreement"
                value={forecastRun ? `${(forecastRun.model_agreement * 100).toFixed(0)}%` : '—'}
                color={forecastRun && forecastRun.model_agreement < 0.5 ? 'red' : 'cyan'}
              />
              <KPICard title="Confidence" value={forecastRun ? `${(forecastRun.confidence * 100).toFixed(0)}%` : '—'} color="purple" />
              <KPICard
                title="Weather Regime"
                value={forecastRun?.regime_detected?.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase()) || '—'}
                color="amber"
              />
              <KPICard
                title="Models Active"
                value={forecastRun ? `${forecastRun.models_used.length}/${forecastRun.models_used.length + forecastRun.models_unavailable.length}` : '—'}
                color="green"
              />
              <KPICard
                title="Extreme Events"
                value={forecastRun?.hazard_events
                  ? Object.values(forecastRun.hazard_events).flat().length.toString()
                  : '0'}
                color="red"
              />
              <KPICard
                title="Data Quality"
                value={forecastRun ? `${(Object.values(forecastRun.quality_scores).reduce((a: number, b: number) => a + b, 0) / Math.max(Object.keys(forecastRun.quality_scores).length, 1) * 100).toFixed(0)}%` : '—'}
                color="cyan"
              />
            </div>

            {/* Main panels */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Model Weights */}
              <div className="glass-card p-6">
                <h3 className="text-lg font-bold mb-4">📊 Model Trust Panel</h3>
                {forecastRun ? (
                  <div className="space-y-3">
                    {Object.entries(forecastRun.model_weights)
                      .sort(([, a], [, b]) => b - a)
                      .map(([model, weight]) => (
                        <WeightBar
                          key={model}
                          model={model}
                          weight={weight}
                          skill={forecastRun.skill_scores[model] || 0}
                          quality={forecastRun.quality_scores[model] || 0}
                        />
                      ))}
                  </div>
                ) : (
                  <p className="text-gray-500 text-sm">Run a forecast to see model weights</p>
                )}
              </div>

              {/* Why This Forecast */}
              <div className="glass-card p-6">
                <h3 className="text-lg font-bold mb-4">🧠 Why This Forecast?</h3>
                {forecastRun ? (
                  <div className="space-y-4">
                    <pre className="text-xs text-gray-300 whitespace-pre-wrap font-mono bg-black/30 rounded-xl p-4 max-h-[350px] overflow-auto">
                      {forecastRun.explanation}
                    </pre>
                  </div>
                ) : (
                  <p className="text-gray-500 text-sm">Run a forecast to see explanation</p>
                )}
              </div>

              {/* Counterfactuals */}
              <div className="glass-card p-6">
                <h3 className="text-lg font-bold mb-4">🔬 What If Model Removed?</h3>
                {forecastRun?.counterfactuals ? (
                  <div className="space-y-3">
                    <p className="text-sm text-gray-400 mb-3">
                      Current blend: <span className="font-bold text-blue-400">{forecastRun.blended_forecast_mean.toFixed(1)}</span>
                    </p>
                    {Object.entries(forecastRun.counterfactuals).map(([model, cf]: [string, any]) => (
                      <div key={model} className="flex justify-between items-center p-2 rounded-lg bg-white/[0.02]">
                        <span className="text-sm">Without {MODEL_NAMES[model] || model}</span>
                        <div className="text-right">
                          <span className="font-mono text-sm">{cf.blend_without?.toFixed(1)}</span>
                          <span className={`ml-2 text-xs ${cf.difference > 0 ? 'text-red-400' : 'text-green-400'}`}>
                            {cf.difference > 0 ? '+' : ''}{cf.difference?.toFixed(1)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-500 text-sm">Run a forecast to see counterfactuals</p>
                )}
              </div>
            </div>

            {/* Controls */}
            <div className="glass-card p-6">
              <h3 className="text-lg font-bold mb-4">⚙️ Forecast Controls</h3>
              <div className="flex flex-wrap gap-4 items-end">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Variable</label>
                  <select value={selectedVariable} onChange={e => setSelectedVariable(e.target.value)}
                    className="bg-[#1a2235] border border-white/10 rounded-lg px-3 py-2 text-sm">
                    <option value="rainfall">Rainfall</option>
                    <option value="temperature">Temperature</option>
                    <option value="wind_speed">Wind Speed</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Lead Time</label>
                  <select value={selectedLeadTime} onChange={e => setSelectedLeadTime(Number(e.target.value))}
                    className="bg-[#1a2235] border border-white/10 rounded-lg px-3 py-2 text-sm">
                    {[6, 12, 24, 48, 72, 120].map(lt => (
                      <option key={lt} value={lt}>{lt}h</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Time Index</label>
                  <input type="number" min={0} max={89} value={selectedTimeIdx}
                    onChange={e => setSelectedTimeIdx(Number(e.target.value))}
                    className="bg-[#1a2235] border border-white/10 rounded-lg px-3 py-2 text-sm w-20" />
                </div>
                <button onClick={runForecast} disabled={loading}
                  className="px-6 py-2 rounded-xl bg-gradient-to-r from-blue-600 to-blue-500 text-white text-sm font-semibold hover:shadow-lg hover:shadow-blue-500/25 transition-all disabled:opacity-50">
                  {loading ? 'Running...' : 'Run Forecast'}
                </button>
                <button onClick={runEvaluatorMode}
                  className="px-4 py-2 rounded-xl border border-purple-500/30 text-purple-400 text-sm font-medium hover:bg-purple-500/10 transition-all">
                  🎯 Evaluator Mode
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ═══ TAB 1: Forecast ═══ */}
        {activeTab === 1 && (
          <div className="space-y-6">
            <div className="glass-card p-6">
              <h3 className="text-lg font-bold mb-4">🗺️ Blended Forecast — {selectedVariable}</h3>
              <p className="text-sm text-gray-400 mb-4">
                Interactive India forecast map. Blended using RSC-MoE adaptive weights.
              </p>
              {/* SVG India Map with heatmap overlay */}
              <div className="bg-[#0d1117] rounded-xl p-4 h-[500px] flex items-center justify-center relative overflow-hidden">
                <IndiaSVGMap forecastRun={forecastRun} variable={selectedVariable} />
              </div>
            </div>

            {/* Model comparison charts */}
            {forecastRun && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="glass-card p-6">
                  <h4 className="font-bold mb-4">Model Weight Distribution</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={weightChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                      <XAxis dataKey="model" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                      <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} />
                      <Tooltip contentStyle={{ background: '#1a2235', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
                      <Bar dataKey="weight" name="Weight %" radius={[4, 4, 0, 0]}>
                        {weightChartData.map((entry: any, i: number) => (
                          <Cell key={i} fill={entry.fill} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="glass-card p-6">
                  <h4 className="font-bold mb-4">Model Skill Radar</h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <RadarChart data={radarData}>
                      <PolarGrid stroke="#1f2937" />
                      <PolarAngleAxis dataKey="subject" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                      <PolarRadiusAxis tick={{ fill: '#6b7280', fontSize: 10 }} />
                      <Radar name="Weight" dataKey="Weight" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.2} />
                      <Radar name="Skill" dataKey="Skill" stroke="#22c55e" fill="#22c55e" fillOpacity={0.1} />
                      <Tooltip contentStyle={{ background: '#1a2235', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
                      <Legend />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ═══ TAB 2: Model Weights ═══ */}
        {activeTab === 2 && (
          <div className="space-y-6">
            <div className="glass-card p-6">
              <h3 className="text-lg font-bold mb-4">🎯 Model Weight Intelligence</h3>
              <p className="text-sm text-gray-400 mb-4">
                Spatial distribution of model trust. Shows which model SAMYUKTA trusts most at each location.
              </p>
              {/* Weight map visualization */}
              <div className="bg-[#0d1117] rounded-xl p-4 h-[500px] flex items-center justify-center">
                <WeightMapVisualization forecastRun={forecastRun} />
              </div>
            </div>

            {/* Model Removal Experiment */}
            <div className="glass-card p-6">
              <h3 className="text-lg font-bold mb-4">🔬 Model Removal Experiment</h3>
              <p className="text-sm text-gray-400 mb-4">
                Remove a model and see how the blended forecast changes.
              </p>
              <div className="flex flex-wrap gap-3">
                {['ncum_g', 'ncum_r', 'gfs', 'wrf', 'ai_model'].map(mid => (
                  <button key={mid} onClick={() => runModelRemoval(mid)}
                    className="px-4 py-2 rounded-lg border border-white/10 text-sm hover:bg-red-500/10 hover:border-red-500/30 hover:text-red-400 transition-all">
                    Remove {MODEL_NAMES[mid]}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ═══ TAB 3: Extreme Weather ═══ */}
        {activeTab === 3 && (
          <div className="space-y-6">
            <div className="glass-card p-6">
              <h3 className="text-lg font-bold mb-2">⚠️ Extreme Weather Center</h3>
              <p className="text-xs text-gray-500 mb-4">Decision-support guidance only. Not an official warning.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {forecastRun?.hazard_events ? (
                <>
                  <HazardCard type="rainfall" events={forecastRun.hazard_events.rainfall || []} />
                  <HazardCard type="heat_wave" events={forecastRun.hazard_events.heat_wave || []} />
                  <HazardCard type="high_wind" events={forecastRun.hazard_events.high_wind || []} />
                </>
              ) : (
                <div className="col-span-3 text-center text-gray-500 py-12">
                  Run a forecast to detect extreme weather events
                </div>
              )}
            </div>

            {/* Disagreement Alert */}
            {forecastRun && forecastRun.model_agreement < 0.5 && (
              <div className="glass-card p-6 glow-amber border-amber-500/30">
                <div className="flex items-center gap-3 mb-3">
                  <span className="text-2xl">⚠️</span>
                  <h4 className="font-bold text-amber-400">HIGH MODEL DISAGREEMENT</h4>
                </div>
                <p className="text-sm text-gray-300">
                  Models show significant disagreement ({((1 - forecastRun.model_agreement) * 100).toFixed(0)}% divergence).
                  Forecast confidence has been reduced accordingly.
                </p>
              </div>
            )}
          </div>
        )}

        {/* ═══ TAB 4: Verification ═══ */}
        {activeTab === 4 && (
          <div className="space-y-6">
            <div className="glass-card p-6">
              <h3 className="text-lg font-bold mb-4">📈 Verification Lab — Forecast Skill Improvement</h3>
              <button onClick={async () => {
                setLoading(true);
                const r = await apiFetch(`/evaluate?variable=${selectedVariable}&lead_time=${selectedLeadTime}`, { method: 'POST' });
                if (r) setVerification({ improvement_table: r.improvement_table });
                setLoading(false);
              }} className="mb-4 px-4 py-2 rounded-lg bg-blue-600/20 text-blue-400 text-sm hover:bg-blue-600/30 transition-all">
                {loading ? 'Evaluating...' : 'Run Full Evaluation'}
              </button>

              {verification?.improvement_table && (
                <>
                  <div className="overflow-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-white/10">
                          <th className="text-left py-3 px-4 text-gray-400">Method</th>
                          <th className="text-right py-3 px-4 text-gray-400">MAE</th>
                          <th className="text-right py-3 px-4 text-gray-400">RMSE</th>
                          <th className="text-right py-3 px-4 text-gray-400">Bias</th>
                          <th className="text-right py-3 px-4 text-gray-400">Correlation</th>
                          <th className="text-right py-3 px-4 text-gray-400">Skill Score</th>
                          <th className="text-right py-3 px-4 text-gray-400">RMSE Impr. %</th>
                        </tr>
                      </thead>
                      <tbody>
                        {verification.improvement_table.map((row: any, i: number) => (
                          <tr key={i} className={`border-b border-white/5 ${row.method === 'SAMYUKTA' ? 'bg-blue-500/10' : ''}`}>
                            <td className={`py-3 px-4 font-medium ${row.method === 'SAMYUKTA' ? 'text-blue-400 font-bold' : ''}`}>
                              {row.method === 'SAMYUKTA' ? '★ ' : ''}{row.method}
                            </td>
                            <td className="text-right py-3 px-4 font-mono">{row.mae?.toFixed(3)}</td>
                            <td className="text-right py-3 px-4 font-mono">{row.rmse?.toFixed(3)}</td>
                            <td className="text-right py-3 px-4 font-mono">{row.bias?.toFixed(3)}</td>
                            <td className="text-right py-3 px-4 font-mono">{row.correlation?.toFixed(3)}</td>
                            <td className="text-right py-3 px-4 font-mono font-bold">{row.skill_score?.toFixed(3)}</td>
                            <td className={`text-right py-3 px-4 font-mono ${
                              row['rmse_improvement_%'] > 0 ? 'text-green-400' :
                              row['rmse_improvement_%'] < 0 ? 'text-red-400' : ''
                            }`}>
                              {row['rmse_improvement_%'] != null ? `${row['rmse_improvement_%'] > 0 ? '+' : ''}${row['rmse_improvement_%'].toFixed(1)}%` : '—'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Verification chart */}
                  <div className="mt-6">
                    <h4 className="font-bold mb-4">Skill Score Comparison</h4>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={verificationChartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                        <XAxis dataKey="method" tick={{ fill: '#9ca3af', fontSize: 10 }} />
                        <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} />
                        <Tooltip contentStyle={{ background: '#1a2235', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
                        <Bar dataKey="Skill Score" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="Correlation" fill="#22c55e" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {/* ═══ TAB 5: Data Health ═══ */}
        {activeTab === 5 && (
          <div className="space-y-6">
            <div className="glass-card p-6">
              <h3 className="text-lg font-bold mb-4">🔧 Data Health Monitor</h3>
              <button onClick={loadDataHealth} className="mb-4 px-4 py-2 rounded-lg bg-blue-600/20 text-blue-400 text-sm">
                Refresh
              </button>

              {dataHealth?.sources ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(dataHealth.sources).map(([mid, info]: [string, any]) => (
                    <div key={mid} className={`glass-card p-4 ${info.available ? 'glow-green' : 'glow-red'}`}>
                      <div className="flex justify-between items-center mb-3">
                        <span className="font-semibold">{MODEL_NAMES[mid] || mid}</span>
                        <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                          info.available ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                        }`}>
                          {info.status}
                        </span>
                      </div>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Quality Score</span>
                          <span className="font-mono">{(info.quality_score * 100).toFixed(0)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Lead Times</span>
                          <span className="text-xs">{info.lead_times?.join(', ')}h</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Data Mode</span>
                          <span className="text-amber-400 text-xs">{info.data_mode}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500 text-sm">Click Refresh to check data source health</p>
              )}
            </div>
          </div>
        )}

        {/* ═══ TAB 6: Audit ═══ */}
        {activeTab === 6 && (
          <div className="space-y-6">
            <div className="glass-card p-6">
              <h3 className="text-lg font-bold mb-4">📋 Forecast Audit Trail</h3>
              {forecastRun ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <p className="text-gray-500">Run ID</p>
                      <p className="font-mono font-bold">{forecastRun.run_id}</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Timestamp</p>
                      <p className="font-mono">{new Date(forecastRun.timestamp).toLocaleString()}</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Processing Time</p>
                      <p className="font-mono">{forecastRun.processing_time_ms.toFixed(0)} ms</p>
                    </div>
                    <div>
                      <p className="text-gray-500">Data Mode</p>
                      <p className="text-amber-400">{forecastRun.data_mode}</p>
                    </div>
                  </div>

                  <details className="bg-black/30 rounded-xl">
                    <summary className="p-4 cursor-pointer text-sm font-medium">Full Forecast Provenance</summary>
                    <pre className="p-4 text-xs text-gray-300 overflow-auto max-h-[400px] font-mono">
                      {JSON.stringify(forecastRun, null, 2)}
                    </pre>
                  </details>
                </div>
              ) : (
                <p className="text-gray-500 text-sm">No forecast runs to audit yet</p>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-white/5 py-4 text-center text-xs text-gray-600">
        SAMYUKTA — Intelligent Hybrid Weather Forecast Fusion | SIH26081 | Ministry of Earth Sciences / NCMRWF
      </footer>
    </div>
  );
}

// ─── India SVG Map Component ─────────────────
function IndiaSVGMap({ forecastRun, variable }: { forecastRun: any; variable: string }) {
  // Simplified India outline with grid overlay for demo
  const lat_range = [6, 38];
  const lon_range = [68, 98];

  const gridSize = 4; // degrees per cell
  const cells = [];

  for (let lat = lat_range[0]; lat < lat_range[1]; lat += gridSize) {
    for (let lon = lon_range[0]; lon < lon_range[1]; lon += gridSize) {
      // Map to SVG coordinates
      const x = ((lon - lon_range[0]) / (lon_range[1] - lon_range[0])) * 400 + 50;
      const y = ((lat_range[1] - lat) / (lat_range[1] - lat_range[0])) * 450 + 25;
      const w = (gridSize / (lon_range[1] - lon_range[0])) * 400;
      const h = (gridSize / (lat_range[1] - lat_range[0])) * 450;

      // Generate color based on forecast data
      let value = 0;
      if (forecastRun) {
        // Use blended forecast mean + spatial variation
        const base = forecastRun.blended_forecast_mean || 0;
        const noise = Math.sin(lat * 0.5) * Math.cos(lon * 0.3) * base * 0.3;
        value = Math.max(0, base + noise);
      }

      let color = getHeatColor(value, variable);
      cells.push({ x, y, w, h, lat, lon, value, color });
    }
  }

  return (
    <svg viewBox="0 0 500 500" className="w-full h-full max-w-md mx-auto">
      {/* Background */}
      <rect x="0" y="0" width="500" height="500" fill="#0d1117" />

      {/* Grid cells */}
      {cells.map((cell, i) => (
        <rect
          key={i}
          x={cell.x} y={cell.y}
          width={cell.w} height={cell.h}
          fill={cell.color}
          opacity={0.7}
          stroke="#1a2235"
          strokeWidth={0.5}
        >
          <title>{`${cell.lat}°N, ${cell.lon}°E: ${cell.value.toFixed(1)}`}</title>
        </rect>
      ))}

      {/* India outline (simplified) */}
      <path
        d="M200,30 L260,50 L290,80 L310,120 L320,160 L300,200 L310,250 L290,300 L260,350 L240,400 L220,430 L200,450 L180,430 L160,400 L140,360 L130,300 L120,250 L110,200 L120,160 L130,120 L150,80 L170,50 Z"
        fill="none"
        stroke="rgba(255,255,255,0.3)"
        strokeWidth={2}
      />

      {/* Labels */}
      <text x="250" y="490" textAnchor="middle" fill="#6b7280" fontSize="11">
        India — {variable.replace('_', ' ')} Forecast
      </text>

      {/* Legend */}
      <defs>
        <linearGradient id="legendGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#22c55e" />
          <stop offset="50%" stopColor="#f59e0b" />
          <stop offset="100%" stopColor="#ef4444" />
        </linearGradient>
      </defs>
      <rect x="350" y="430" width="120" height="8" rx="4" fill="url(#legendGrad)" />
      <text x="350" y="455" fill="#6b7280" fontSize="8">Low</text>
      <text x="470" y="455" textAnchor="end" fill="#6b7280" fontSize="8">High</text>
    </svg>
  );
}

function getHeatColor(value: number, variable: string): string {
  let normalized = 0;
  if (variable === 'rainfall') {
    normalized = Math.min(value / 100, 1);
  } else if (variable === 'temperature') {
    normalized = Math.min(Math.max((value - 15) / 35, 0), 1);
  } else {
    normalized = Math.min(value / 30, 1);
  }

  const r = Math.round(34 + normalized * 205);
  const g = Math.round(197 - normalized * 130);
  const b = Math.round(94 - normalized * 26);
  return `rgb(${r},${g},${b})`;
}

// ─── Weight Map Visualization ────────────────
function WeightMapVisualization({ forecastRun }: { forecastRun: any }) {
  if (!forecastRun?.model_weights) {
    return <p className="text-gray-500 text-sm">Run a forecast to see weight maps</p>;
  }

  const weights = forecastRun.model_weights;
  const sortedModels = Object.entries(weights).sort(([, a]: any, [, b]: any) => b - a);
  const total = Object.values(weights).reduce((a: any, b: any) => a + b, 0);

  // Create pie data
  const pieData = sortedModels.map(([model, weight]: [string, any]) => ({
    name: MODEL_NAMES[model] || model,
    value: Math.round(weight * 100),
    color: MODEL_COLORS[model] || '#6b7280',
  }));

  return (
    <div className="w-full h-full flex items-center justify-center gap-8">
      <ResponsiveContainer width="50%" height={400}>
        <PieChart>
          <Pie
            data={pieData}
            cx="50%" cy="50%"
            innerRadius={80}
            outerRadius={140}
            paddingAngle={3}
            dataKey="value"
          >
            {pieData.map((entry, i) => (
              <Cell key={i} fill={entry.color} stroke="#0a0e1a" strokeWidth={2} />
            ))}
          </Pie>
          <Tooltip contentStyle={{ background: '#1a2235', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
          <Legend />
        </PieChart>
      </ResponsiveContainer>

      <div className="space-y-3">
        <h4 className="text-sm font-bold text-gray-400">Dominant Model by Region</h4>
        {sortedModels.slice(0, 3).map(([model, weight]: [string, any]) => (
          <div key={model} className="flex items-center gap-3">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: MODEL_COLORS[model] }} />
            <span className="text-sm">{MODEL_NAMES[model]}</span>
            <span className="font-mono text-sm font-bold" style={{ color: MODEL_COLORS[model] }}>
              {Math.round(weight * 100)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
