"""
SAMYUKTA — Complete Demo Runner

Runs the full end-to-end demo scenario:
1. Generate demo data (if not exists)
2. Initialize pipeline
3. Execute forecast blending
4. Run verification
5. Print results

Usage:
    python scripts/run_demo.py
"""
import sys
import json
import logging
from pathlib import Path

# Fix Windows console encoding for Unicode/emojis
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SAMYUKTA-DEMO")


def main():
    print("=" * 70)
    print("  SAMYUKTA — Intelligent Hybrid Weather Forecast Fusion")
    print("  SIH26081 Demo Run")
    print("  ⚠ All data is DEMO / SYNTHETIC — NOT real observations")
    print("=" * 70)
    print()

    # 1. Check/generate demo data
    demo_dir = ROOT / "data" / "demo"
    if not (demo_dir / "ground_truth.nc").exists():
        logger.info("Demo data not found, generating...")
        from ml.ingestion.synthetic_generator import generate_demo_dataset
        generate_demo_dataset(
            output_dir=str(demo_dir),
            n_days=90,
            start_date="2024-06-01",
            resolution=0.5,
        )

    # 2. Initialize pipeline
    from ml.inference.pipeline import SamyuktaPipeline
    pipeline = SamyuktaPipeline(data_dir=str(demo_dir))
    status = pipeline.initialize()
    print(f"\n✅ Pipeline initialized: {status.get('steps', [])[-1] if status.get('steps') else 'unknown'}")

    # 3. Run demo scenario
    print("\n🚀 Running demo scenario...\n")
    demo_result = pipeline.run_demo_scenario()

    # 4. Print results
    print("\n" + "=" * 70)
    print("  DEMO RESULTS")
    print("=" * 70)

    if demo_result.get("primary_run"):
        run = demo_result["primary_run"]
        print(f"\n📍 Primary Run ID: {run.get('run_id')}")
        print(f"📅 Variable: {run.get('variable')}")
        print(f"⏱  Lead Time: {run.get('lead_time_hours')}h")
        print(f"🌦  Regime: {run.get('regime_detected', '').replace('_', ' ').title()}")
        print(f"🎯 Confidence: {run.get('confidence', 0):.1%}")
        print(f"📊 Model Agreement: {run.get('model_agreement', 0):.1%}")
        print(f"🌡  Blended Forecast Mean: {run.get('blended_forecast_mean', 0):.2f}")
        print(f"📏 Uncertainty: ±{run.get('uncertainty', 0):.2f}")

        print("\n📊 Model Weights:")
        for mid, w in sorted(run.get("model_weights", {}).items(), key=lambda x: -x[1]):
            bar = "█" * int(w * 40)
            print(f"   {mid:12s}: {w:.1%}  {bar}")

        print("\n🔬 Counterfactuals (What if model removed?):")
        for mid, cf in run.get("counterfactuals", {}).items():
            diff = cf.get("difference", 0)
            sign = "+" if diff > 0 else ""
            print(f"   Without {mid:12s}: {cf.get('blend_without', 0):.2f}  ({sign}{diff:.2f})")

    # 5. Verification
    if demo_result.get("improvement_table"):
        print("\n📈 Verification — Forecast Skill Improvement:")
        print(f"   {'Method':<25s} {'MAE':>8s} {'RMSE':>8s} {'Skill':>8s} {'Impr.%':>8s}")
        print("   " + "-" * 60)
        for row in demo_result["improvement_table"]:
            impr = row.get("rmse_improvement_%", "")
            impr_str = f"{impr:+.1f}%" if isinstance(impr, (int, float)) else "—"
            print(f"   {row['method']:<25s} {row.get('mae', 0):>8.3f} {row.get('rmse', 0):>8.3f} "
                  f"{row.get('skill_score', 0):>8.3f} {impr_str:>8s}")

    print("\n" + "=" * 70)
    print("  ✅ Demo complete. Start the dashboard:")
    print("     Backend:  python -m uvicorn backend.app.main:app --reload")
    print("     Frontend: cd frontend && npm install && npm run dev")
    print("=" * 70)


if __name__ == "__main__":
    main()
