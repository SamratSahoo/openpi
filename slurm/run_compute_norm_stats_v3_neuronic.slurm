#!/bin/bash
#SBATCH --job-name=openpi-norm-stats-v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:1
#SBATCH --partition=all
#SBATCH --account=seas
#SBATCH --time=02:00:00
#SBATCH --output=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/norm_stats_v3_%j.out
#SBATCH --error=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/norm_stats_v3_%j.err
#
# Neuronic equivalent of run_compute_norm_stats.slurm (Della), targeting the EXTENDED v3 config
# (DROID-100 + corrected toys-no-collision-v2) instead of v2/droid100-only.
#
# README step 7.4: compute norm stats into assets/pi05droid-full-d100+toys/ BEFORE training.
# Run after the merge job has produced SamratSahoo/d100_toys20.

set -euo pipefail

export UV_CACHE_DIR=/n/fs/tamp-vla/.cache/uv
export HF_HOME=/n/fs/tamp-vla/.cache/huggingface
export HF_HUB_CACHE=/n/fs/tamp-vla/.cache/huggingface/hub
export HF_LEROBOT_HOME=/n/fs/tamp-vla/.cache/huggingface/lerobot
export OPENPI_DATA_HOME=/n/fs/tamp-vla/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /n/fs/tamp-vla/tamp-vla/openpi

echo "=== Computing norm stats for pi05droid-full-d100+toys on $(hostname) at $(date) ==="
uv run --frozen scripts/compute_norm_stats.py --config-name pi05droid-full-d100+toys
echo "=== Done at $(date) ==="
