#!/bin/bash
#SBATCH --job-name=openpi-upload-ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=all
#SBATCH --account=seas
#SBATCH --exclude=neu306
#SBATCH --time=06:00:00
#SBATCH --output=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/upload_ckpt_%x_%j.out
#SBATCH --error=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/upload_ckpt_%x_%j.err
# Upload a trained checkpoint dir to a HF model repo. Args: <checkpoint_dir> <repo_id>
set -euo pipefail
export UV_CACHE_DIR=/n/fs/tamp-vla/.cache/uv
export HF_HOME=/n/fs/tamp-vla/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
cd /n/fs/tamp-vla/tamp-vla/openpi
echo "=== $(hostname) $(date) | upload $1 -> $2 ==="
exec .venv/bin/python -u slurm/upload_ckpt.py "$1" "$2"
