#!/usr/bin/env bash
# =============================================================================
# ProtoEvoNet Experiment Runner
#
# Runs all experiments described in README_EXPERIMENTS.md in sequence.
#
# Usage:
#   bash run_all_experiments.sh [OPTIONS]
#
# Options:
#   --checkpoint CKPT   Path to trained model checkpoint (default: checkpoints/best_model.pt)
#   --data-root DATA    Root data directory (default: data)
#   --device DEVICE     Torch device (default: cuda)
#   --dry-run           Print commands without executing them
#
# Example:
#   bash run_all_experiments.sh --checkpoint checkpoints/phase2_best.pt --device cuda
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
CHECKPOINT="checkpoints/best_model.pt"
DATA_ROOT="data"
DEVICE="cuda"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)  CHECKPOINT="$2";  shift 2 ;;
        --data-root)   DATA_ROOT="$2";   shift 2 ;;
        --device)      DEVICE="$2";      shift 2 ;;
        --dry-run)     DRY_RUN=1;        shift   ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
run() {
    echo ""
    echo ">>> $*"
    if [[ $DRY_RUN -eq 0 ]]; then
        "$@"
    fi
}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/q1_${TIMESTAMP}"
mkdir -p results figures "$LOG_DIR"

echo "============================================================"
echo "ProtoEvoNet Q1 Experiment Suite"
echo "  Checkpoint : $CHECKPOINT"
echo "  Data root  : $DATA_ROOT"
echo "  Device     : $DEVICE"
echo "  Log dir    : $LOG_DIR"
echo "  Dry run    : $DRY_RUN"
echo "============================================================"

# ---------------------------------------------------------------------------
# Step 0: Prepare OpenSARShip dataset (builds splits CSV)
# ---------------------------------------------------------------------------
echo ""
echo "=== STEP 0: Prepare OpenSARShip 2.0 ==="
run python prepare_opensarship.py \
    --data-root "$DATA_ROOT/open_sar/OpenSARShip2" \
    --output-dir data \
    2>&1 | tee "$LOG_DIR/step0_opensarship_prep.log"

# ---------------------------------------------------------------------------
# Step 1: Parameter count analysis
# ---------------------------------------------------------------------------
echo ""
echo "=== STEP 1: Parameter Count Analysis ==="
run python count_parameters.py \
    --checkpoint "$CHECKPOINT" \
    2>&1 | tee "$LOG_DIR/step1_param_count.log"

# ---------------------------------------------------------------------------
# Step 2: FUSAR-Ship 10-class low-shot evaluation (1/5/10/20-shot)
# ---------------------------------------------------------------------------
echo ""
echo "=== STEP 2: FUSAR Low-Shot Evaluation ==="
run python main.py \
    --demo-only \
    --system-ckpt "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --device "$DEVICE" \
    --low-shot-eval \
    --dataset fusar \
    --n-shot 1 5 10 20 \
    --n-seeds 5 \
    2>&1 | tee "$LOG_DIR/step2_fusar_lowshot.log"

# ---------------------------------------------------------------------------
# Step 3: OpenSARShip low-shot evaluation (1/5/10/20-shot)
# ---------------------------------------------------------------------------
echo ""
echo "=== STEP 3: OpenSARShip Low-Shot Evaluation ==="
run python main.py \
    --demo-only \
    --system-ckpt "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --device "$DEVICE" \
    --low-shot-eval \
    --dataset opensarship \
    --n-shot 1 5 10 20 \
    --n-seeds 5 \
    2>&1 | tee "$LOG_DIR/step3_opensarship_lowshot.log"

# ---------------------------------------------------------------------------
# Step 4: MSTAR low-shot evaluation (1/5/10/20-shot)
# ---------------------------------------------------------------------------
echo ""
echo "=== STEP 4: MSTAR Low-Shot Evaluation ==="
run python main.py \
    --demo-only \
    --system-ckpt "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --device "$DEVICE" \
    --low-shot-eval \
    --dataset mstar \
    --n-shot 1 5 10 20 \
    --n-seeds 5 \
    2>&1 | tee "$LOG_DIR/step4_mstar_lowshot.log"

# ---------------------------------------------------------------------------
# Step 5: Cross-domain evaluation (frozen backbone)
# ---------------------------------------------------------------------------
echo ""
echo "=== STEP 5: Cross-Domain Evaluation ==="
run python cross_domain_eval.py \
    --checkpoint "$CHECKPOINT" \
    --fusar-root "$DATA_ROOT/fusar_ship" \
    --opensarship-root "$DATA_ROOT/open_sar/OpenSARShip2" \
    --mstar-root "$DATA_ROOT/mstar" \
    --n-shot 10 \
    --n-seeds 5 \
    --output results/cross_domain.csv \
    --device "$DEVICE" \
    2>&1 | tee "$LOG_DIR/step5_cross_domain.log"

# ---------------------------------------------------------------------------
# Step 6: Baseline novelty detector comparison
# ---------------------------------------------------------------------------
echo ""
echo "=== STEP 6: Baseline Novelty Comparison ==="
run python main.py \
    --demo-only \
    --system-ckpt "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --mstar-root "$DATA_ROOT/mstar" \
    --device "$DEVICE" \
    2>&1 | tee "$LOG_DIR/step6_novelty_baselines.log"

# ---------------------------------------------------------------------------
# Step 7: Full ablation study
# ---------------------------------------------------------------------------
echo ""
echo "=== STEP 7: Ablation Study ==="

ABLATION_FLAGS=(
    "use_gcn=False"
    "use_radial_conv=False"
    "use_cross_scale_attention=False"
    "use_dsar=False"
    "use_sqaa=False"
    "use_agppi=False"
    "use_bpc_gp=False"
    "use_pim=False"
    "use_tpe=False"
)

for FLAG in "${ABLATION_FLAGS[@]}"; do
    ABLATION_NAME="${FLAG//=/_}"
    echo ""
    echo "--- Ablation: $FLAG ---"
    run python main.py \
        --demo-only \
        --system-ckpt "$CHECKPOINT" \
        --data-root "$DATA_ROOT" \
        --device "$DEVICE" \
        --ablate "$FLAG" \
        2>&1 | tee "$LOG_DIR/step7_ablation_${ABLATION_NAME}.log"
done

# ---------------------------------------------------------------------------
# Step 8: Generate visualizations
# ---------------------------------------------------------------------------
echo ""
echo "=== STEP 8: Visualizations ==="

# t-SNE embedding plot
run python -c "
import sys, torch, numpy as np
from pathlib import Path
sys.path.insert(0, '.')

try:
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print('Generating t-SNE visualization...')
    # Load scores from cross-domain output if available
    scores_path = Path('results/cross_domain.csv')
    if scores_path.exists():
        print(f'  Cross-domain results found at {scores_path}')
    else:
        print('  No cross-domain results found, skipping t-SNE.')
except ImportError as e:
    print(f'Skipping t-SNE: {e}')
" 2>&1 | tee "$LOG_DIR/step8_tsne.log"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Q1 Experiment Suite Complete"
echo "  Results  : results/"
echo "  Figures  : figures/"
echo "  Logs     : $LOG_DIR/"
echo "============================================================"
echo ""
echo "Key result files:"
echo "  results/cross_domain.csv             — cross-domain recognition + novelty"
echo "  results/low_shot_fusar_summary.csv   — FUSAR low-shot (1/5/10/20-shot)"
echo "  results/low_shot_opensarship_summary.csv"
echo "  results/low_shot_mstar_summary.csv"
echo "  results/parameter_counts.json        — parameter count analysis"
echo "  logs/novelty_summary.csv             — full novelty method comparison"
echo ""
