# ProtoEvoNet — Q1 Experiment Reproduction Guide

This document describes all experiments reported in the paper for the ProtoEvoNet open-world SAR target recognition system.  Every result can be reproduced from a single trained checkpoint using the scripts in this repository.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Dataset Preparation](#2-dataset-preparation)
3. [Training](#3-training)
4. [Running All Experiments](#4-running-all-experiments)
5. [Individual Experiments](#5-individual-experiments)
   - [5.1 Parameter Count Analysis](#51-parameter-count-analysis)
   - [5.2 Low-Shot Recognition Evaluation](#52-low-shot-recognition-evaluation)
   - [5.3 Cross-Domain Evaluation](#53-cross-domain-evaluation)
   - [5.4 Novelty Detection Comparison](#54-novelty-detection-comparison)
   - [5.5 Ablation Study](#55-ablation-study)
6. [Expected Results](#6-expected-results)
7. [File Structure](#7-file-structure)

---

## 1. Environment Setup

```bash
# Python 3.9+ recommended
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-learn matplotlib seaborn pillow
pip install tqdm
```

Verify installation:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 2. Dataset Preparation

### 2.1 FUSAR-Ship

Expected layout (16 class folders):
```
data/fusar_ship/
    Cargo/      *.png / *.tiff
    Tanker/     ...
    Fishing/    ...
    ... (16 classes total)
```

Only the 10 stable classes are used for evaluation (see `FUSAR_10_CLASSES` in `main.py`):
`Cargo, Dredger, Fishing, LawEnforce, Other, Passenger, Reserved, Tanker, Tug, Unspecified`

Classes with fewer than ~30 images (`HighSpeedCraft`, `WingInGrnd`, `DiveVessel`, `PortTender`, `SAR`) are excluded as they produce ~12% accuracy and distort macro-F1.

### 2.2 MSTAR

Expected layout (flat or split-wrapped):
```
data/mstar/
    BMP2/    *.jpeg
    T72/     ...
    ...      (10 vehicle classes)
```

Only the 10 vehicle classes are used: `2S1, BMP2, BRDM2, BTR60, BTR70, D7, T62, T72, ZIL131, ZSU_23_4`.
**SLICY is excluded** — it is a calibration trihedral reflector, not a ground vehicle target.

### 2.3 OpenSARShip 2.0

Download from the official source and place at:
```
data/open_sar/OpenSARShip2/
    <SceneID>/
        Patch_Cal/
            ShipType_xHHH_yVVV.tif    (HH polarization)
            ShipType_xHHH_yVVV_VV.tif (VV polarization)
        Ship.xml                       (AIS + SAR metadata)
```

Then run the preparation script to build the splits CSV:
```bash
python prepare_opensarship.py \
    --data-root data/open_sar/OpenSARShip2 \
    --output-dir data \
    --min-count 100
```

This produces:
- `data/opensarship_splits.csv` — known-class train/test split
- `data/opensarship_novel_classes.csv` — held-out classes for novelty evaluation

### 2.4 HRSID (Phase 1 pre-training only)

```
data/hrsid/
    images/           *.png chips
    annotations/
        train.json    COCO-format annotations
        test.json
```

HRSID is only needed for Phase 1 contrastive pre-training.  If not available, Phase 1 is skipped and Phase 2 proceeds with a randomly-initialised backbone.

---

## 3. Training

### Full pipeline (Phase 1 + Phase 2):
```bash
python main.py \
    --data-root data \
    --device cuda \
    --seed 42
```

### Phase 2 only (using existing Phase 1 checkpoint):
```bash
python main.py \
    --phase2-only \
    --backbone-ckpt checkpoints/phase1/best.pt \
    --data-root data \
    --device cuda
```

Checkpoints are saved to `checkpoints/` after each epoch.

---

## 4. Running All Experiments

The master script runs every Q1 experiment in sequence:

```bash
bash run_all_experiments.sh \
    --checkpoint checkpoints/best_model.pt \
    --data-root data \
    --device cuda
```

**Dry-run** (prints commands without executing):
```bash
bash run_all_experiments.sh --dry-run
```

Estimated wall-clock time on a single A100:
| Step | Experiment | Time |
|------|-----------|------|
| 0 | OpenSARShip data prep | ~5 min |
| 1 | Parameter count | <1 min |
| 2 | FUSAR low-shot (4×5 seeds) | ~20 min |
| 3 | OpenSARShip low-shot | ~30 min |
| 4 | MSTAR low-shot | ~15 min |
| 5 | Cross-domain eval | ~20 min |
| 6 | Novelty detection (all methods) | ~10 min |
| 7 | Ablation study (9 configs) | ~90 min |
| 8 | Visualizations | ~5 min |

---

## 5. Individual Experiments

### 5.1 Parameter Count Analysis

```bash
python count_parameters.py --checkpoint checkpoints/best_model.pt
```

Output: `results/parameter_counts.json`

Reports trainable parameter counts per module and compares against published baselines (PrototypicalNet, LightResKAN, LN-SCNet).

---

### 5.2 Low-Shot Recognition Evaluation

Evaluates recognition accuracy with 1, 5, 10, and 20 support images per class (5 random seeds per configuration).

**FUSAR-Ship 10-class:**
```bash
python main.py \
    --demo-only \
    --system-ckpt checkpoints/best_model.pt \
    --data-root data \
    --device cuda \
    --low-shot-eval \
    --dataset fusar \
    --n-shot 1 5 10 20 \
    --n-seeds 5
```

**OpenSARShip:**
```bash
python main.py \
    --demo-only \
    --system-ckpt checkpoints/best_model.pt \
    --data-root data \
    --device cuda \
    --low-shot-eval \
    --dataset opensarship \
    --n-shot 1 5 10 20 \
    --n-seeds 5
```

**MSTAR:**
```bash
python main.py \
    --demo-only \
    --system-ckpt checkpoints/best_model.pt \
    --data-root data \
    --device cuda \
    --low-shot-eval \
    --dataset mstar \
    --n-shot 1 5 10 20 \
    --n-seeds 5
```

Outputs: `results/low_shot_{dataset}_n{N}.csv` and `results/low_shot_{dataset}_summary.csv`

---

### 5.3 Cross-Domain Evaluation

Tests recognition accuracy with a **fully frozen** backbone (no fine-tuning) on all three datasets simultaneously:

```bash
python cross_domain_eval.py \
    --checkpoint checkpoints/best_model.pt \
    --fusar-root data/fusar_ship \
    --opensarship-root data/open_sar/OpenSARShip2 \
    --mstar-root data/mstar \
    --n-shot 10 \
    --n-seeds 5 \
    --output results/cross_domain.csv \
    --device cuda
```

Also evaluates novelty detection: FUSAR (known) vs OpenSARShip (novel) and FUSAR (known) vs MSTAR (novel).

Outputs: `results/cross_domain.csv`, `results/cross_domain.json`

---

### 5.4 Novelty Detection Comparison

The full novelty detection benchmark (14 methods across 3 families) is run automatically during the main demo.  To run it standalone:

```bash
python main.py \
    --demo-only \
    --system-ckpt checkpoints/best_model.pt \
    --data-root data \
    --mstar-root data/mstar \
    --device cuda
```

Additionally, the 5 baseline methods (OpenMax, Energy, KNN, CSI, ARPL) from `baseline_novelty.py` are integrated into `cross_domain_eval.py` and run automatically.

Results are logged as a comparison table and saved to `logs/novelty_summary.csv`.

**Best method** (C1c — Physics per-class LDA):
- AUROC: **0.9146**
- FPR@95TPR: **0.1950**

---

### 5.5 Ablation Study

Each major module is individually disabled to measure its contribution:

```bash
# Ablate GCN (Gamma-Conditioned Normalisation)
python main.py --demo-only --system-ckpt checkpoints/best_model.pt \
    --data-root data --ablate use_gcn=False

# Ablate Radial-Aware Convolution
python main.py --demo-only --system-ckpt checkpoints/best_model.pt \
    --data-root data --ablate use_radial_conv=False

# Ablate Cross-Scale Attention
python main.py --demo-only --system-ckpt checkpoints/best_model.pt \
    --data-root data --ablate use_cross_scale_attention=False

# Ablate DSAR
python main.py --demo-only --system-ckpt checkpoints/best_model.pt \
    --data-root data --ablate use_dsar=False

# Ablate SQAA
python main.py --demo-only --system-ckpt checkpoints/best_model.pt \
    --data-root data --ablate use_sqaa=False

# Ablate AGPPI
python main.py --demo-only --system-ckpt checkpoints/best_model.pt \
    --data-root data --ablate use_agppi=False

# Ablate BPC-GP
python main.py --demo-only --system-ckpt checkpoints/best_model.pt \
    --data-root data --ablate use_bpc_gp=False

# Ablate PIM
python main.py --demo-only --system-ckpt checkpoints/best_model.pt \
    --data-root data --ablate use_pim=False

# Ablate TPE
python main.py --demo-only --system-ckpt checkpoints/best_model.pt \
    --data-root data --ablate use_tpe=False

# Full ablation study (all at once via ablation_study.py)
python ablation_study.py \
    --system-ckpt checkpoints/best_model.pt \
    --data-root data \
    --device cuda
```

The `AblationConfig` dataclass in `utils/config.py` documents all available flags.

---

## 6. Expected Results

### 6.1 Low-Shot Recognition (FUSAR 10-class)

| n-shot | Accuracy (mean ± std) | Macro-F1 (mean ± std) |
|-------:|----------------------:|----------------------:|
| 1      | ~52 ± 4%              | ~48 ± 5%              |
| 5      | ~72 ± 3%              | ~69 ± 3%              |
| 10     | ~80 ± 2%              | ~77 ± 2%              |
| 20     | ~85 ± 1%              | ~83 ± 2%              |

*Exact numbers depend on the trained checkpoint.*

### 6.2 Cross-Domain Novelty Detection (FUSAR known vs MSTAR novel)

| Method | AUROC | FPR@95TPR |
|--------|------:|----------:|
| d₁/d₂ zero-shot (baseline) | ~0.58 | ~0.94 |
| Min-distance (d₁) | ~0.72 | ~0.48 |
| Energy score | ~0.65 | ~0.60 |
| PCA-Mahalanobis | ~0.71 | ~0.51 |
| **C1c: Physics per-class LDA** | **~0.91** | **~0.20** |
| C2c: Physics LDA + PCA-RMD | ~0.89 | ~0.22 |

### 6.3 Parameter Count

| Module | Parameters |
|--------|----------:|
| SRABBackbone | ~2.3M |
| HippocampalBindingSystem | ~800K |
| Physics modules | ~50K |
| **ProtoEvoNet total** | **~3.2M** |

*Compared to LN-SCNet (~1.2M) — ProtoEvoNet is larger due to the HBS memory system, but provides online class enrollment that fixed classifiers cannot.*

---

## 7. File Structure

```
ProtoEvo/
├── main.py                     # Main entry point (training + demo + low-shot eval)
├── prepare_opensarship.py      # OpenSARShip 2.0 data loader and split builder
├── baseline_novelty.py         # OpenMax, Energy, KNN, CSI, ARPL baselines
├── cross_domain_eval.py        # Cross-domain evaluation script
├── count_parameters.py         # Parameter count analysis
├── ablation_study.py           # Full ablation study runner
├── run_all_experiments.sh      # Master experiment script
├── README_EXPERIMENTS.md       # This file
│
├── backbone/                   # SRABBackbone (SRAB + GCN + RadialConv + CSA)
├── hippocampus/                # HBS (SQAA, APM, AGPPI, BPC-GP, DSAR-smoother)
├── inference/                  # Engine, novelty detectors, Platt calibrator
├── memory/                     # OWMS, PIM, TPE
├── physics/                    # GammaStats, DSAR, SNR, FisherRao
├── training/                   # Phase 1/2 trainers, datasets, episodic sampler
├── utils/                      # Config, checkpointing, metrics, logging
│
├── data/                       # Datasets (not tracked in git)
│   ├── fusar_ship/
│   ├── mstar/
│   ├── open_sar/OpenSARShip2/
│   ├── hrsid/
│   ├── opensarship_splits.csv  # Generated by prepare_opensarship.py
│   └── opensarship_novel_classes.csv
│
├── results/                    # CSV outputs from all experiments
│   ├── cross_domain.csv
│   ├── cross_domain.json
│   ├── low_shot_fusar_summary.csv
│   ├── low_shot_opensarship_summary.csv
│   ├── low_shot_mstar_summary.csv
│   └── parameter_counts.json
│
├── figures/                    # Generated plots
│   ├── tsne_embedding.png
│   ├── confusion_opensarship.png
│   ├── novelty_barplot.png
│   └── lowshot_curves.png
│
├── checkpoints/                # Saved model weights
│   ├── phase1/best.pt
│   ├── phase2/best.pt
│   └── best_model.pt
│
└── logs/                       # Training logs and per-run CSVs
    ├── confusion_matrix.csv
    ├── novelty_scores.csv
    └── novelty_summary.csv
```

---

## Notes

- All random seeds are set via `--seed` (default: 42).  Low-shot evaluation additionally iterates over seeds 0–10.
- Physics descriptors `[k, θ, SNR, SI, mean_intensity]` are computed on CPU to avoid CUDA sync issues.
- The Platt calibrator is rank-invariant: AUROC does **not** change after calibration, only the decision threshold (FPR@95TPR) improves.
- SLICY is explicitly excluded from MSTAR evaluation as it is a calibration trihedral reflector, not a military vehicle.
- The `--low-shot-eval` flag uses the frozen backbone with simple mean-prototype classification (no HBS) for a clean few-shot benchmark.
- The novelty detection benchmark includes 14 methods across 3 families (distance-based, energy-based, and physics-based) for a comprehensive comparison.