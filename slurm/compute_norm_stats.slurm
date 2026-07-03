#!/bin/bash
#SBATCH --job-name=openpi-norm-stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:1
#SBATCH --partition=all
#SBATCH --account=seas
#SBATCH --time=04:00:00
#SBATCH --output=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/norm_stats_%j.out
#SBATCH --error=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/norm_stats_%j.err
#
# Compute normalization stats for ANY openpi TrainConfig, on Neuronic (1x L40).
#
#   Usage:  sbatch compute_norm_stats.slurm <config_name> [extra compute_norm_stats.py args]
#   e.g.:   sbatch compute_norm_stats.slurm pi05base-full-d100+toys100sim-stream
#           sbatch compute_norm_stats.slurm pi05base-droid+toys100sim-stream --max-frames 200000
#
# Writes assets/<config_name>/<asset_id>/norm_stats.*. Run this BEFORE run_training.slurm for the
# same config. (For big streaming configs like full DROID, pass --max-frames to subsample.)

set -euo pipefail

CONFIG="${1:?usage: sbatch compute_norm_stats.slurm <config_name> [extra args]}"
shift

# Caches on the 1 TB project FS (home is only ~16 GB on Neuronic).
export UV_CACHE_DIR=/n/fs/tamp-vla/.cache/uv
export HF_HOME=/n/fs/tamp-vla/.cache/huggingface
export HF_HUB_CACHE=/n/fs/tamp-vla/.cache/huggingface/hub
export HF_LEROBOT_HOME=/n/fs/tamp-vla/.cache/huggingface/lerobot
export OPENPI_DATA_HOME=/n/fs/tamp-vla/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /n/fs/tamp-vla/tamp-vla/openpi

echo "=== Computing norm stats for '$CONFIG' on $(hostname) at $(date) ==="
uv run --frozen scripts/compute_norm_stats.py --config-name "$CONFIG" "$@"
echo "=== Done at $(date) ==="
