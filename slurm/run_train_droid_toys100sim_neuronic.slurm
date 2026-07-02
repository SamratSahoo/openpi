#!/bin/bash
#SBATCH --job-name=openpi-train-droid-toys100sim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:l40:8
#SBATCH --partition=all
#SBATCH --account=seas
#SBATCH --exclude=neu306
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/train_droid_toys100sim_%j.out
#SBATCH --error=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/train_droid_toys100sim_%j.err
#
# FULL fine-tune of the pi0.5-DROID checkpoint on SamratSahoo/toys100_sim (pi05droid-toys100sim).
# 8x L40 / FSDP=8 (model+optimizer sharded 8 ways). This config REUSES the official DROID norm stats
# (assets pulled from gs://openpi-assets/checkpoints/pi05_droid/assets, asset_id=droid), so no
# norm-stats job is needed.

set -euo pipefail

export UV_CACHE_DIR=/n/fs/tamp-vla/.cache/uv
export HF_HOME=/n/fs/tamp-vla/.cache/huggingface
export HF_HUB_CACHE=/n/fs/tamp-vla/.cache/huggingface/hub
export HF_LEROBOT_HOME=/n/fs/tamp-vla/.cache/huggingface/lerobot
export OPENPI_DATA_HOME=/n/fs/tamp-vla/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /n/fs/tamp-vla/tamp-vla/openpi

echo "=== Full fine-tune pi05droid-toys100sim (8x L40, FSDP=8) on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
uv run --frozen scripts/train.py pi05droid-toys100sim \
    --exp-name=toys100sim_from_droid_full_ft \
    --fsdp-devices 8 \
    --overwrite
echo "=== Done at $(date) ==="
