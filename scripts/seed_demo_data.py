"""
SAMYUKTA — Demo Data Seeder

Run this script to generate the complete synthetic demo dataset.
This creates all files needed for the application to run without internet.

Usage:
    python scripts/seed_demo_data.py

Output:
    data/demo/ground_truth.nc
    data/demo/forecast_{model}_lt{hours}.nc
    data/demo/metadata.json
"""
import sys
import logging
from pathlib import Path

# Fix Windows console encoding for Unicode/emojis
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ml.ingestion.synthetic_generator import generate_demo_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main():
    output_dir = ROOT / "data" / "demo"
    print("=" * 60)
    print("  SAMYUKTA — Synthetic Demo Data Generator")
    print("  ⚠ All data is DEMO / SYNTHETIC — NOT real observations")
    print("=" * 60)
    print()

    files = generate_demo_dataset(
        output_dir=str(output_dir),
        n_days=90,            # 3 months covering monsoon season
        start_date="2024-06-01",
        resolution=0.5,       # 0.5° grid for fast demo
        lat_range=(6.0, 38.0),
        lon_range=(68.0, 98.0),
        lead_times=[6, 12, 24, 48, 72, 120],
        model_ids=["ncum_g", "ncum_r", "gfs", "wrf", "ai_model"],
    )

    print()
    print(f"✅ Generated {len(files)} files in {output_dir}")
    for name, path in files.items():
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"   {name}: {path.name} ({size_mb:.1f} MB)")

    print()
    print("Ready for demo. Start the backend with:")
    print("   python -m uvicorn backend.app.main:app --reload")
    print()


if __name__ == "__main__":
    main()
