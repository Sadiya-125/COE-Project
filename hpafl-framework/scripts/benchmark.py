"""Benchmark script — runs all training pipelines and generates a full report.

Executes in sequence: centralized → fedavg → hpafl, then calls generate_full_report().
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import HAPFLConfig, KAGGLE_DATA_ROOT, KAGGLE_OUTPUT_ROOT, is_kaggle
from evaluation.evaluate import generate_full_report
from scripts.run_centralized import run_centralized
from scripts.run_fedavg import run_fedavg
from scripts.run_hpafl import run_hpafl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the complete HPAFL benchmark suite."""
    cfg = HAPFLConfig()
    if is_kaggle():
        cfg.data_root = KAGGLE_DATA_ROOT
        cfg.output_root = KAGGLE_OUTPUT_ROOT
    else:
        _root = Path(__file__).resolve().parents[1]
        cfg.data_root = str(_root.parent / "archive")
        cfg.output_root = str(_root)

    print("\n" + "=" * 60)
    print("HPAFL BENCHMARK SUITE")
    print("=" * 60)

    stages = [
        ("Centralised Baseline", run_centralized),
        ("FedAvg Baseline", run_fedavg),
        ("HPAFL Full System", run_hpafl),
    ]

    timings = {}
    for name, fn in stages:
        print(f"\n>>> Running: {name}")
        t0 = time.time()
        try:
            fn(cfg)
            elapsed = time.time() - t0
            timings[name] = elapsed
            print(f"    Completed in {elapsed:.1f}s")
        except Exception as exc:
            logger.error("Stage '%s' failed: %s", name, exc)
            timings[name] = -1

    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    for name, t in timings.items():
        status = f"{t:.1f}s" if t >= 0 else "FAILED"
        print(f"  {name:<30} {status}")

    print("\n>>> Generating full evaluation report...")
    generate_full_report(cfg.results_dir)


if __name__ == "__main__":
    main()
