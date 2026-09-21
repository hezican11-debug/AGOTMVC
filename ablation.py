"""
Run the complete Cora ablation suite once and archive each JSON result.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "run_pipeline.py"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "ablation_results"

DATASET = "Cora"
SOURCE_JSON = MODELS_DIR / f"{DATASET}_results.json"

EXPERIMENTS = [
    # (
    #     "Full",
    #     [
    #         "--gb_generation", "adaptive",
    #         "--matching", "ot",
    #         "--positive_mode", "multi",
    #         "--use_uncertainty", "1",
    #         "--gb_drift_weight", "0.05",
    #     ],
    # ),
    (
        "AGB",
        [
            "--gb_generation", "kmeans",
            "--matching", "ot",
            "--positive_mode", "multi",
            "--use_uncertainty", "1",
            "--gb_drift_weight", "0.05",
        ],
    ),
    (
        "OT",
        [
            "--gb_generation", "adaptive",
            "--matching", "hard",
            "--positive_mode", "multi",
            "--use_uncertainty", "1",
            "--gb_drift_weight", "0.05",
        ],
    ),
    (
        "WMP",
        [
            "--gb_generation", "adaptive",
            "--matching", "ot",
            "--positive_mode", "single",
            "--use_uncertainty", "1",
            "--gb_drift_weight", "0.05",
        ],
    ),
    (
        "UNC",
        [
            "--gb_generation", "adaptive",
            "--matching", "ot",
            "--positive_mode", "multi",
            "--use_uncertainty", "0",
            "--gb_drift_weight", "0.05",
        ],
    ),
    (
        "drift",
        [
            "--gb_generation", "adaptive",
            "--matching", "ot",
            "--positive_mode", "multi",
            "--use_uncertainty", "1",
            "--gb_drift_weight", "0",
        ],
    ),
]


def load_multi_metrics(path: Path) -> tuple[float | None, float | None]:
    """Read ACC/PUR from metrics.multi for a compact console summary."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        multi = payload.get("metrics", {}).get("multi", {})
        return multi.get("ACC"), multi.get("PUR")
    except Exception:
        return None, None


def main() -> int:
    if not PIPELINE.exists():
        print(f"[FATAL] Cannot find: {PIPELINE}")
        print("Place this script in the same directory as run_pipeline.py.")
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print(f"Dataset : {DATASET}")
    print(f"Runs    : {len(EXPERIMENTS)}")
    print(f"Output  : {OUTPUT_DIR}")
    print("=" * 88)

    completed = []

    for idx, (suffix, extra_args) in enumerate(EXPERIMENTS, start=1):
        target_json = OUTPUT_DIR / f"{DATASET}_{suffix}.json"

        if SOURCE_JSON.exists():
            SOURCE_JSON.unlink()

        command = [
            sys.executable,
            "-u",
            str(PIPELINE),
            "--only",
            DATASET,
            *extra_args,
        ]

        print("\n" + "#" * 88)
        print(f"[{idx}/{len(EXPERIMENTS)}] START: {DATASET}_{suffix}")
        print("Command:")
        print(" ".join(f'"{x}"' if " " in x else x for x in command))
        print("#" * 88)

        start = time.time()
        proc = subprocess.run(command, cwd=str(ROOT))
        elapsed = time.time() - start

        if proc.returncode != 0:
            print(f"\n[FAILED] {DATASET}_{suffix}")
            print(f"Exit code: {proc.returncode}")
            print("Already completed JSON files are kept.")
            return proc.returncode or 1

        if not SOURCE_JSON.exists():
            print(f"\n[FAILED] Pipeline returned success, but result JSON is missing:")
            print(SOURCE_JSON)
            print("Already completed JSON files are kept.")
            return 3

        shutil.copy2(SOURCE_JSON, target_json)

        acc, pur = load_multi_metrics(target_json)
        metric_text = ""
        if acc is not None and pur is not None:
            metric_text = f" | multi ACC={float(acc):.4f}, PUR={float(pur):.4f}"

        print(
            f"[DONE] {target_json.name}"
            f" | {elapsed / 60:.1f} min"
            f"{metric_text}"
        )
        completed.append(target_json)

    print("\n" + "=" * 88)
    print("ALL ABLATION RUNS FINISHED")
    for path in completed:
        print(f"  - {path.relative_to(ROOT)}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
