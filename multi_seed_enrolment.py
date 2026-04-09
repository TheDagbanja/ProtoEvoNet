#!/usr/bin/env python3
"""
Multi‑seed evaluation of incremental enrolment.

Runs the enrolment experiment for a given dataset and number of shots,
repeating over N random seeds, and reports mean ± std accuracy.
"""

import argparse
import csv
import logging
import random
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

from eval_enrolment import (
    load_system_with_shape_filtering,
    evaluate_enrolment,
    MSTAREnrolmentDataset,
)
from prepare_opensarship import OpenSARShipDataset
from training.datasets import MSTARDataset
from utils.config import ProtoEvoNetConfig

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["opensarship", "mstar"])
    parser.add_argument("--system-ckpt", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-shot", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    all_accs = []
    per_class_accs = defaultdict(list)

    for seed in range(args.seeds):
        set_seed(seed)
        cfg = ProtoEvoNetConfig()
        cfg.device = str(device)
        system = load_system_with_shape_filtering(args.system_ckpt, cfg, device)
        engine = system["engine"]

        if args.dataset == "opensarship":
            splits_csv = Path(args.data_root) / "opensarship_splits.csv"
            if not splits_csv.exists():
                logger.error("OpenSARShip splits CSV not found.")
                sys.exit(1)
            dataset = OpenSARShipDataset(
                csv_path=str(splits_csv),
                split="all",
                image_size=(128, 128),
                polarization="VV",
                normalize=True,
                stats_csv=str(Path(args.data_root) / "opensarship_stats_VV.csv")
            )
        else:
            mstar_root = Path(args.data_root) / "mstar"
            full = MSTARDataset(root=str(mstar_root), image_size=(128, 128))
            from main import MSTAR_VEHICLE_CLASSES
            present = [c for c in MSTAR_VEHICLE_CLASSES if c in full.classes]
            dataset = MSTAREnrolmentDataset(full, class_filter=present)

        overall_acc, per_class = evaluate_enrolment(
            engine, dataset, f"{args.dataset}_seed{seed}",
            n_enrol_per_class=args.n_shot,
            results_dir=f"results/multi_seed/{args.dataset}_shot{args.n_shot}"
        )
        all_accs.append(overall_acc)
        for c, acc in per_class.items():
            per_class_accs[c].append(acc)

    mean_acc = np.mean(all_accs)
    std_acc = np.std(all_accs)
    print(f"\n=== {args.dataset.upper()} | {args.n_shot}-shot | seeds={args.seeds} ===")
    print(f"Overall accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print("Per‑class accuracy (mean ± std):")
    for c, acc_list in per_class_accs.items():
        m = np.mean(acc_list)
        s = np.std(acc_list)
        print(f"  {c:12s} {m:5.1f}% ± {s:4.1f}%")

    # Save summary CSV
    out_csv = Path(f"results/multi_seed/{args.dataset}_shot{args.n_shot}_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "mean_acc_%", "std_acc_%"])
        for c in sorted(per_class_accs.keys()):
            writer.writerow([c, f"{np.mean(per_class_accs[c]):.2f}", f"{np.std(per_class_accs[c]):.2f}"])
        writer.writerow(["OVERALL", f"{mean_acc:.2f}", f"{std_acc:.2f}"])


if __name__ == "__main__":
    main()