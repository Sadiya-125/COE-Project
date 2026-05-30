"""HAM10000 dataset partitioning into non-IID hospital splits.

Implements Dirichlet-based label-skew partitioning with intentional secondary
skew to simulate realistic multi-hospital heterogeneity as described in the
HPAFL framework design.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is on sys.path when this script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import CLASS_NAMES, DX_TO_CLASS, HOSPITAL_IDS, HAPFLConfig

logger = logging.getLogger(__name__)


class DatasetPartitioner:
    """Partitions HAM10000 into non-IID hospital splits using Dirichlet sampling.

    Steps:
        1. Load metadata CSV and map dx labels to integer indices.
        2. Apply Dirichlet non-IID split per class.
        3. Apply secondary skew to reinforce hospital specialisation.
        4. Save indices as JSON and print distribution statistics.

    Args:
        config: HPAFL configuration object.
        seed: Random seed for reproducibility.
    """

    # Secondary skew specification: hospital -> preferred classes (class indices)
    _SECONDARY_SKEW: Dict[str, List[int]] = {
        "A": [0, 1],  # MEL, NV
        "B": [2, 3],  # BCC, AK
        "C": [5, 6],  # DF, VASC
    }
    _SWAP_FRACTION: float = 0.15

    def __init__(self, config: HAPFLConfig, seed: int = 42) -> None:
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # Partitions are written to output_root (writable), not data_root (read-only on Kaggle)
        self.partition_dir = config.partition_dir
        self.partition_dir.mkdir(parents=True, exist_ok=True)

    def load_metadata(self) -> pd.DataFrame:
        """Load HAM10000 metadata CSV and encode dx labels as integers.

        Returns:
            DataFrame with added 'label' column (0-6) and 'class_code' column.

        Raises:
            FileNotFoundError: If metadata CSV is not found.
        """
        csv_path = Path(self.config.data_root) / "HAM10000_metadata.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        df["class_code"] = df["dx"].map(DX_TO_CLASS)
        df["label"] = df["class_code"].map(
            {name: idx for idx, name in enumerate(CLASS_NAMES)}
        )
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        logger.info(
            "Loaded %d samples with %d classes from %s",
            len(df),
            df["label"].nunique(),
            csv_path,
        )
        return df

    def dirichlet_split(
        self, labels: np.ndarray
    ) -> Dict[str, List[int]]:
        """Partition sample indices using per-class Dirichlet sampling.

        Args:
            labels: Integer class label for each sample.

        Returns:
            Mapping from hospital ID to list of sample indices.
        """
        n_hospitals = self.config.num_hospitals
        alpha = self.config.dirichlet_alpha
        splits: Dict[str, List[int]] = {h: [] for h in HOSPITAL_IDS[:n_hospitals]}

        for cls in range(self.config.num_classes):
            cls_indices = np.where(labels == cls)[0].tolist()
            if not cls_indices:
                continue
            self.rng.shuffle(cls_indices)

            # Dirichlet proportions for this class across hospitals
            proportions = self.rng.dirichlet(
                alpha * np.ones(n_hospitals)
            )
            # Enforce hospital_splits as a soft prior by blending
            prior = np.array(self.config.hospital_splits[:n_hospitals])
            proportions = 0.7 * proportions + 0.3 * prior
            proportions = proportions / proportions.sum()

            boundaries = (np.cumsum(proportions) * len(cls_indices)).astype(int)
            boundaries = np.clip(boundaries, 0, len(cls_indices))
            starts = np.concatenate([[0], boundaries[:-1]])

            for idx, hosp_id in enumerate(HOSPITAL_IDS[:n_hospitals]):
                chunk = cls_indices[starts[idx]: boundaries[idx]]
                splits[hosp_id].extend(chunk)

        return splits

    def apply_secondary_skew(
        self, splits: Dict[str, List[int]], labels: np.ndarray
    ) -> Dict[str, List[int]]:
        """Reinforce hospital specialisation by swapping 15% of samples.

        Hospitals preferring certain classes gain samples of those classes
        from other hospitals, creating a stronger label-distribution skew.

        Args:
            splits: Current per-hospital index lists.
            labels: Integer class label for each sample.

        Returns:
            Updated splits with secondary skew applied.
        """
        n_hospitals = len(HOSPITAL_IDS[: self.config.num_hospitals])
        hosp_ids = HOSPITAL_IDS[:n_hospitals]

        for hosp_id in hosp_ids:
            preferred_classes = self._SECONDARY_SKEW.get(hosp_id, [])
            other_hospitals = [h for h in hosp_ids if h != hosp_id]

            for other_id in other_hospitals:
                other_indices = np.array(splits[other_id])
                if len(other_indices) == 0:
                    continue

                other_labels = labels[other_indices]
                preferred_mask = np.isin(other_labels, preferred_classes)
                preferred_in_other = other_indices[preferred_mask]

                n_swap = int(len(preferred_in_other) * self._SWAP_FRACTION)
                if n_swap == 0:
                    continue

                swap_idx = self.rng.choice(
                    len(preferred_in_other), size=n_swap, replace=False
                )
                to_swap = preferred_in_other[swap_idx]

                # Move from other → self
                splits[hosp_id].extend(to_swap.tolist())
                keep_mask = np.ones(len(splits[other_id]), dtype=bool)
                to_swap_set = set(to_swap.tolist())
                splits[other_id] = [
                    i for i in splits[other_id] if i not in to_swap_set
                ]

        return splits

    def partition(self) -> Dict[str, List[int]]:
        """Run the full partitioning pipeline.

        Returns:
            Mapping from hospital ID to list of sample indices.
        """
        df = self.load_metadata()
        labels = df["label"].values

        logger.info("Running Dirichlet non-IID split (alpha=%.2f)…", self.config.dirichlet_alpha)
        splits = self.dirichlet_split(labels)

        logger.info("Applying secondary skew (swap_fraction=%.0f%%)…", self._SWAP_FRACTION * 100)
        splits = self.apply_secondary_skew(splits, labels)

        self._save_splits(splits, df)
        self._print_stats(splits, labels)
        return splits

    def _save_splits(self, splits: Dict[str, List[int]], df: pd.DataFrame) -> None:
        """Persist partition indices to JSON files.

        Args:
            splits: Per-hospital sample indices.
            df: Full metadata DataFrame.
        """
        for hosp_id, indices in splits.items():
            path = self.partition_dir / f"hospital_{hosp_id}_indices.json"
            with open(path, "w") as f:
                json.dump(sorted(indices), f)
            logger.info("Saved %d indices → %s", len(indices), path)

        # Also save image_id lists for convenience
        labels = df["label"].values
        stats: Dict[str, Dict] = {}
        for hosp_id, indices in splits.items():
            hosp_labels = labels[indices]
            class_counts = {
                CLASS_NAMES[c]: int(np.sum(hosp_labels == c))
                for c in range(self.config.num_classes)
            }
            stats[hosp_id] = {
                "total": len(indices),
                "class_distribution": class_counts,
            }

        stats_path = self.partition_dir / "partition_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Saved partition statistics → %s", stats_path)

    def _print_stats(
        self, splits: Dict[str, List[int]], labels: np.ndarray
    ) -> None:
        """Print class distribution per hospital to stdout.

        Args:
            splits: Per-hospital sample indices.
            labels: Integer class labels for all samples.
        """
        header = f"{'Hospital':<12}" + "".join(f"{c:<8}" for c in CLASS_NAMES) + f"{'Total':<8}"
        print("\n" + "=" * len(header))
        print("PARTITION STATISTICS")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        for hosp_id, indices in splits.items():
            if not indices:
                continue
            hosp_labels = labels[indices]
            counts = [int(np.sum(hosp_labels == c)) for c in range(self.config.num_classes)]
            row = f"{'Hospital ' + hosp_id:<12}" + "".join(f"{c:<8}" for c in counts) + f"{len(indices):<8}"
            print(row)

        all_counts = [
            int(np.sum(labels == c)) for c in range(self.config.num_classes)
        ]
        print("-" * len(header))
        total_row = f"{'Total':<12}" + "".join(f"{c:<8}" for c in all_counts) + f"{len(labels):<8}"
        print(total_row)
        print("=" * len(header) + "\n")

        # Imbalance ratio
        for hosp_id, indices in splits.items():
            if not indices:
                continue
            hosp_labels = labels[indices]
            counts = [np.sum(hosp_labels == c) for c in range(self.config.num_classes)]
            nonzero = [c for c in counts if c > 0]
            if nonzero:
                ratio = max(nonzero) / min(nonzero)
                logger.info("Hospital %s imbalance ratio: %.1f:1", hosp_id, ratio)


def verify_partitions(config: HAPFLConfig) -> bool:
    """Verify that saved partition files have no overlapping indices.

    Args:
        config: HPAFL configuration object.

    Returns:
        True if partitions are valid (no overlap), False otherwise.
    """
    partition_dir = config.partition_dir
    all_indices: Dict[str, set] = {}

    for hosp_id in HOSPITAL_IDS[: config.num_hospitals]:
        path = partition_dir / f"hospital_{hosp_id}_indices.json"
        if not path.exists():
            logger.error("Missing partition file: %s", path)
            return False
        with open(path) as f:
            all_indices[hosp_id] = set(json.load(f))

    # Check pairwise overlap
    valid = True
    hospital_list = list(all_indices.keys())
    for i in range(len(hospital_list)):
        for j in range(i + 1, len(hospital_list)):
            h1, h2 = hospital_list[i], hospital_list[j]
            overlap = all_indices[h1] & all_indices[h2]
            if overlap:
                logger.error(
                    "Overlap between Hospital %s and %s: %d samples", h1, h2, len(overlap)
                )
                valid = False
            else:
                logger.info("Hospital %s ∩ Hospital %s = ∅ ✓", h1, h2)

    if valid:
        logger.info("Partition verification PASSED — no overlapping indices.")
    return valid


if __name__ == "__main__":
    from config import is_kaggle, KAGGLE_DATA_ROOT, KAGGLE_OUTPUT_ROOT
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = HAPFLConfig()
    if is_kaggle():
        cfg.data_root = KAGGLE_DATA_ROOT
        cfg.output_root = KAGGLE_OUTPUT_ROOT
    else:
        _project_root = Path(__file__).resolve().parents[1]
        cfg.data_root = str(_project_root.parent / "archive")
        cfg.output_root = str(_project_root)
    partitioner = DatasetPartitioner(cfg)
    partitioner.partition()
    verify_partitions(cfg)
