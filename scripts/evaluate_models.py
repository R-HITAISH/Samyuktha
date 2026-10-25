"""
SAMYUKTA — Model Evaluation & Training Script

Runs the complete evaluation pipeline comparing baselines against SAMYUKTA.
Generates improvement tables and ablation results.

Usage:
    python scripts/evaluate_models.py
"""
import sys
import json
import logging
from pathlib import Path
import numpy as np

# Fix Windows console encoding for Unicode/emojis
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    from ml.inference.pipeline import SamyuktaPipeline
    from ml.evaluation.verification import VerificationFramework, AblationStudy

    demo_dir = ROOT / "data" / "demo"

    # Check data
    if not (demo_dir / "ground_truth.nc").exists():
        print("Demo data not found. Run: python scripts/seed_demo_data.py")
        return

    # Initialize pipeline
    pipeline = SamyuktaPipeline(data_dir=str(demo_dir))
    pipeline.initialize()

    print("=" * 70)
    print("  SAMYUKTA — Comprehensive Model Evaluation")
    print("  ⚠ Using DEMO / SYNTHETIC DATA")
    print("=" * 70)

    # Run evaluation across multiple time steps
    variables = ["rainfall", "temperature", "wind_speed"]
    lead_times = [24, 48, 72]

    for var in variables:
        print(f"\n{'='*50}")
        print(f"  Variable: {var}")
        print(f"{'='*50}")

        for lt in lead_times:
            n_times = min(pipeline.ground_truth.dims.get("time", 0), 30)
            for t in range(0, n_times, 5):
                try:
                    pipeline.run_forecast(time_idx=t, lead_time=lt, variable=var)
                except Exception as e:
                    logging.warning(f"Error at t={t}, lt={lt}, var={var}: {e}")

        # Print improvement table
        table = pipeline.verification.get_improvement_table(var)
        if table:
            print(f"\n  {'Method':<25s} {'RMSE':>8s} {'MAE':>8s} {'Skill':>8s} {'Impr.%':>8s}")
            print("  " + "-" * 55)
            for row in table:
                impr = row.get("rmse_improvement_%", "")
                impr_str = f"{impr:+.1f}%" if isinstance(impr, (int, float)) else "—"
                marker = "★ " if row["method"] == "SAMYUKTA" else "  "
                print(f"  {marker}{row['method']:<23s} {row.get('rmse', 0):>8.3f} "
                      f"{row.get('mae', 0):>8.3f} {row.get('skill_score', 0):>8.3f} {impr_str:>8s}")

    # Save results
    output_path = ROOT / "data" / "evaluation_results.json"
    pipeline.verification.save(str(output_path))
    print(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
