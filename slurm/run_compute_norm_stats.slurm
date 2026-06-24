#!/bin/bash
#SBATCH --job-name=openpi-norm-stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --time=01:00:00
#SBATCH --account=tsilver
#SBATCH --output=/scratch/gpfs/TSILVER/ss1824/openpi/slurm/logs/norm_stats_%j.out
#SBATCH --error=/scratch/gpfs/TSILVER/ss1824/openpi/slurm/logs/norm_stats_%j.err

export UV_CACHE_DIR=/scratch/gpfs/TSILVER/ss1824/.cache/uv
export HF_HOME=/scratch/gpfs/TSILVER/ss1824/.cache/huggingface
export HF_LEROBOT_HOME=/scratch/gpfs/TSILVER/ss1824/.cache/huggingface/lerobot
export OPENPI_DATA_HOME=/scratch/gpfs/TSILVER/ss1824/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /scratch/gpfs/TSILVER/ss1824/openpi

module load proxy/default
echo "=== Computing norm stats for pi05droid-full-d100 on $(hostname) at $(date) ==="
uv run --frozen scripts/compute_norm_stats.py --config-name pi05droid-full-d100
echo "=== Done at $(date) ==="
