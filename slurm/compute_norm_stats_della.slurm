#!/bin/bash
#SBATCH --job-name=openpi-norm-stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# No --partition: Della auto-routes GPU jobs and rejects an explicit --partition (see training script).
#SBATCH --gres=gpu:a100:1
#SBATCH --account=tsilver
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/TSILVER/ss1824/tamp-vla/openpi/slurm/logs/norm_stats_%j.out
#SBATCH --error=/scratch/gpfs/TSILVER/ss1824/tamp-vla/openpi/slurm/logs/norm_stats_%j.err
#
# Compute normalization stats for ANY openpi TrainConfig, on Della (1x A100). Della twin of
# compute_norm_stats.slurm. No gpu80 constraint here -- computing stats is light and fits on any
# A100, so this stays off the (scarcer) 80 GB nodes that training needs.
#
#   Usage:  sbatch compute_norm_stats_della.slurm <config_name> [extra compute_norm_stats.py args]
#   e.g.:   sbatch compute_norm_stats_della.slurm pi05_droid100_lerobot
#           sbatch compute_norm_stats_della.slurm pi05_droid100_lerobot --max-frames 200000
#
# Writes assets/<config_name>/<asset_id>/norm_stats.*. Run this BEFORE run_training_della.slurm for
# the same config. (For big streaming configs, pass --max-frames to subsample.)

set -euo pipefail

CONFIG="${1:?usage: sbatch compute_norm_stats_della.slurm <config_name> [extra args]}"
shift

# Caches on the 5 TB /scratch/gpfs fileset (home is only ~50 GB on Della).
export UV_CACHE_DIR=/scratch/gpfs/TSILVER/ss1824/.cache/uv
export HF_HOME=/scratch/gpfs/TSILVER/ss1824/.cache/huggingface
export HF_HUB_CACHE=/scratch/gpfs/TSILVER/ss1824/.cache/huggingface/hub
export HF_LEROBOT_HOME=/scratch/gpfs/TSILVER/ss1824/.cache/huggingface/lerobot
export OPENPI_DATA_HOME=/scratch/gpfs/TSILVER/ss1824/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /scratch/gpfs/TSILVER/ss1824/tamp-vla/openpi

echo "=== Computing norm stats for '$CONFIG' on $(hostname) at $(date) ==="
uv run --frozen scripts/compute_norm_stats.py --config-name "$CONFIG" "$@"
echo "=== Done at $(date) ==="
