#!/bin/bash
#SBATCH --job-name=openpi-norm-stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:1
#SBATCH --partition=all
#SBATCH --account=seas
#SBATCH --time=02:00:00
#SBATCH --output=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/norm_stats_%j.out
#SBATCH --error=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/norm_stats_%j.err
#
# Neuronic equivalent of run_compute_norm_stats.slurm (Della).
# Differences vs Della: shared /n/fs caches, no proxy module, GPU is `gpu:l40:1` (not a100),
# and it targets the EXTENDED v2 config (DROID-100 + toys) instead of droid100-only.
#
# README step 7.4: compute norm stats into assets/pi05_droid100_extended_v2_lerobot/ BEFORE training.
# Run after the merge job has produced samratsahoo/droid_100_extended_v2.

set -euo pipefail

export UV_CACHE_DIR=/n/fs/tamp-vla/.cache/uv
export HF_HOME=/n/fs/tamp-vla/.cache/huggingface
export HF_HUB_CACHE=/n/fs/tamp-vla/.cache/huggingface/hub
export HF_LEROBOT_HOME=/n/fs/tamp-vla/.cache/huggingface/lerobot
export OPENPI_DATA_HOME=/n/fs/tamp-vla/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /n/fs/tamp-vla/tamp-vla/openpi

echo "=== Computing norm stats for pi05_droid100_extended_v2_lerobot on $(hostname) at $(date) ==="
uv run --frozen scripts/compute_norm_stats.py --config-name pi05_droid100_extended_v2_lerobot
echo "=== Done at $(date) ==="
