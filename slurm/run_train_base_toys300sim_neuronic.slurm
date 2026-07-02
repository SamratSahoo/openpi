#!/bin/bash
#SBATCH --job-name=openpi-train-base-toys300sim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:l40:8
#SBATCH --partition=all
#SBATCH --account=seas
#SBATCH --exclude=neu306
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/train_base_toys300sim_%j.out
#SBATCH --error=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/train_base_toys300sim_%j.err
#
# FULL fine-tune of the *base* pi0.5 checkpoint on SamratSahoo/toys300_sim (pi05base-toys300sim).
# 8x L40 / FSDP=8 (full FT needs >70 GB; an L40 is ~46 GB, so model+optimizer is sharded 8 ways).
# Prereq: run_compute_norm_stats_base_toys300sim_neuronic.slurm must finish first (fresh norm stats
# in assets/pi05base-toys300sim/; we do NOT reuse the official DROID stats here).

set -euo pipefail

export UV_CACHE_DIR=/n/fs/tamp-vla/.cache/uv
export HF_HOME=/n/fs/tamp-vla/.cache/huggingface
export HF_HUB_CACHE=/n/fs/tamp-vla/.cache/huggingface/hub
export HF_LEROBOT_HOME=/n/fs/tamp-vla/.cache/huggingface/lerobot
export OPENPI_DATA_HOME=/n/fs/tamp-vla/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /n/fs/tamp-vla/tamp-vla/openpi

echo "=== Full fine-tune pi05base-toys300sim (8x L40, FSDP=8) on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
uv run --frozen scripts/train.py pi05base-toys300sim \
    --exp-name=toys300sim_from_base_full_ft \
    --fsdp-devices 8 \
    --overwrite
echo "=== Done at $(date) ==="
