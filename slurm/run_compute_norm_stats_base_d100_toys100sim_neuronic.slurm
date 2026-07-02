#!/bin/bash
#SBATCH --job-name=openpi-norm-base-d100-toys100sim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:1
#SBATCH --partition=all
#SBATCH --account=seas
#SBATCH --time=02:00:00
#SBATCH --output=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/norm_base_d100_toys100sim_%j.out
#SBATCH --error=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/norm_base_d100_toys100sim_%j.err
#
# Compute norm stats for pi05base-full-d100+toys100sim (dataset SamratSahoo/d100_toys100_sim, from HF).
# This config fine-tunes from the *base* pi0.5 checkpoint and does NOT reuse the official DROID
# norm stats, so fresh stats must be computed into assets/pi05base-full-d100+toys100sim/ before training.

set -euo pipefail

export UV_CACHE_DIR=/n/fs/tamp-vla/.cache/uv
export HF_HOME=/n/fs/tamp-vla/.cache/huggingface
export HF_HUB_CACHE=/n/fs/tamp-vla/.cache/huggingface/hub
export HF_LEROBOT_HOME=/n/fs/tamp-vla/.cache/huggingface/lerobot
export OPENPI_DATA_HOME=/n/fs/tamp-vla/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /n/fs/tamp-vla/tamp-vla/openpi

echo "=== Computing norm stats for pi05base-full-d100+toys100sim on $(hostname) at $(date) ==="
uv run --frozen scripts/compute_norm_stats.py --config-name pi05base-full-d100+toys100sim
echo "=== Done at $(date) ==="
