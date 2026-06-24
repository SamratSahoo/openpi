#!/bin/bash
#SBATCH --job-name=openpi-train-droid100-base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:l40:8
#SBATCH --partition=all
#SBATCH --account=seas
# neu306 reproducibly faults this run with CUDA_ERROR_ILLEGAL_ADDRESS on the first step, while the
# identical fsdp=8 step runs fine on other 8x L40 nodes -> exclude it (suspected bad GPU/node; report
# to cluster admins). Drop this line once neu306 is fixed.
#SBATCH --exclude=neu306
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/train_droid100_base_%j.out
#SBATCH --error=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/train_droid100_base_%j.err
#
# FULL fine-tune of the *base* pi0.5 checkpoint (pi05_base, NOT pi05_droid) on DROID-100 (joint schema).
# Same 8x L40 / FSDP=8 setup as run_train_v3_neuronic.slurm; see that script for the FSDP rationale.
#
# Prereq: run run_compute_norm_stats_droid100_base_neuronic.slurm first so the fresh norm stats exist in
# assets/pi05base-full-d100/ (we do NOT reuse the official DROID stats here). The pi05_base
# checkpoint is pulled from gs://openpi-assets into $OPENPI_DATA_HOME on first run.

set -euo pipefail

export UV_CACHE_DIR=/n/fs/tamp-vla/.cache/uv
export HF_HOME=/n/fs/tamp-vla/.cache/huggingface
export HF_HUB_CACHE=/n/fs/tamp-vla/.cache/huggingface/hub
export HF_LEROBOT_HOME=/n/fs/tamp-vla/.cache/huggingface/lerobot
export OPENPI_DATA_HOME=/n/fs/tamp-vla/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /n/fs/tamp-vla/tamp-vla/openpi

echo "=== Full fine-tune pi05base-full-d100 (8x L40, FSDP=8) on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
uv run --frozen scripts/train.py pi05base-full-d100 \
    --exp-name=droid100_from_base_full_ft \
    --fsdp-devices 8 \
    --overwrite
echo "=== Done at $(date) ==="
