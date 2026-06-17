#!/bin/bash
#SBATCH --job-name=openpi-train-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:4
#SBATCH --partition=all
#SBATCH --account=seas
#SBATCH --time=00:30:00
#SBATCH --output=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/train_test_%j.out
#SBATCH --error=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/train_test_%j.err
#
# Neuronic equivalent of run_train_test.slurm (Della): the train.py unit test as a quick
# multi-GPU smoke check. Differences vs Della: shared /n/fs caches, gpu:l40:4 (no a100).

set -euo pipefail

export UV_CACHE_DIR=/n/fs/tamp-vla/.cache/uv
export HF_HOME=/n/fs/tamp-vla/.cache/huggingface
export HF_HUB_CACHE=/n/fs/tamp-vla/.cache/huggingface/hub
export HF_LEROBOT_HOME=/n/fs/tamp-vla/.cache/huggingface/lerobot
export OPENPI_DATA_HOME=/n/fs/tamp-vla/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /n/fs/tamp-vla/tamp-vla/openpi

echo "=== Running train unit test on $(hostname) at $(date) ==="
uv run --frozen pytest scripts/train_test.py -v
echo "=== Done at $(date) ==="
