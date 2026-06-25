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
#SBATCH --output=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/norm_stats_%x_%j.out
#SBATCH --error=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/norm_stats_%x_%j.err
#
# Generic norm-stats job: compute normalization stats for any config whose data does NOT reuse the
# official DROID norm stats (i.e. the pi05_base fine-tunes, which need their own stats in
# assets/<config>/ BEFORE training). Pass the config name as the first positional arg:
#
#   sbatch --job-name=<config> slurm/run_compute_norm_stats_neuronic.slurm <config>

set -euo pipefail

CONFIG="${1:?Usage: sbatch run_compute_norm_stats_neuronic.slurm <config-name>}"

export UV_CACHE_DIR=/n/fs/tamp-vla/.cache/uv
export HF_HOME=/n/fs/tamp-vla/.cache/huggingface
export HF_HUB_CACHE=/n/fs/tamp-vla/.cache/huggingface/hub
export HF_LEROBOT_HOME=/n/fs/tamp-vla/.cache/huggingface/lerobot
export OPENPI_DATA_HOME=/n/fs/tamp-vla/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /n/fs/tamp-vla/tamp-vla/openpi

echo "=== Computing norm stats for ${CONFIG} on $(hostname) at $(date) ==="
uv run --frozen scripts/compute_norm_stats.py --config-name "${CONFIG}"
echo "=== Done at $(date) ==="
