#!/usr/bin/env python3
"""
Multi‑seed in‑domain FUSAR evaluation with random support splits.
Usage: python fusar_multi_seed.py <n_classes> <n_shots>
Example: python fusar_multi_seed.py 5 5
"""

import subprocess
import sys
import re
import numpy as np

def parse_accuracy(log_text):
    match = re.search(r"Overall recognition accuracy:\s+(\d+\.\d+)%", log_text)
    return float(match.group(1)) if match else None

def main():
    if len(sys.argv) < 3:
        print("Usage: python fusar_multi_seed.py <n_classes> <n_shots>")
        sys.exit(1)
    n_classes = sys.argv[1]
    n_shots = sys.argv[2]
    seeds = 5
    accuracies = []
    for seed in range(seeds):
        cmd = [
            "python", "main.py", "--demo-only",
            "--system-ckpt", "checkpoints/phase2/best.pt",
            "--data-root", "data",
            "--mstar-root", "data/mstar",
            "--fusar-n-classes", n_classes,
            "--n-enrol", n_shots,
            "--seed", str(seed),
            "--random-support"
        ]
        print(f"Running seed {seed} (classes={n_classes}, shots={n_shots})...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        acc = parse_accuracy(result.stdout)
        if acc is not None:
            accuracies.append(acc)
            print(f"Seed {seed} accuracy: {acc:.2f}%")
        else:
            print(f"Seed {seed} failed.")
    if accuracies:
        print(f"\n=== Results for {n_classes}-class, {n_shots}-shot ===")
        print(f"Mean accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
        print(f"Individual seeds: {', '.join(f'{a:.2f}' for a in accuracies)}")

if __name__ == "__main__":
    main()