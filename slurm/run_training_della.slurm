#!/bin/bash
#SBATCH --job-name=openpi-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# No --partition: Della's submit plugin auto-routes GPU jobs by gres/constraint/time and REJECTS
# an explicit --partition=gpu. --constraint=gpu80 is what pins this to an 80 GB A100.
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --account=tsilver
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/gpfs/TSILVER/ss1824/tamp-vla/openpi/slurm/logs/train_%j.out
#SBATCH --error=/scratch/gpfs/TSILVER/ss1824/tamp-vla/openpi/slurm/logs/train_%j.err
#
# Full fine-tune ANY openpi TrainConfig on Della (1x A100-80GB, --fsdp-devices 1). This is the
# Della twin of run_training.slurm (which is 8x L40 + FSDP=8 on Neuronic); slurm/server.py picks
# whichever matches the cluster it is launched on.
#
# Mirrors W&B run llo47spj (droid100_joint_ft): a single 80 GB card holds the whole pi0.5 model +
# AdamW state (~78 GB, ~90% of an 80 GB A100), global batch 32, so nothing is sharded and
# --fsdp-devices is 1. --constraint=gpu80 keeps the job off Della's 40 GB A100s, where full FT
# would OOM. Della's gres name is a plain `a100` (no l40-style quirk).
#
#   Usage:  sbatch run_training_della.slurm <config_name> [exp_name] [extra train.py args]
#   e.g.:   sbatch run_training_della.slurm pi05_droid100_lerobot
#           sbatch run_training_della.slurm pi05_droid100_lerobot my_run --num-train-steps 30000
#
# exp_name defaults to the config name. Run compute_norm_stats_della.slurm for the same config
# first (unless it inherits a checkpoint's stats). W&B logs online once you've run `wandb login`.

set -euo pipefail

CONFIG="${1:?usage: sbatch run_training_della.slurm <config_name> [exp_name] [extra args]}"
shift
# Optional positional exp_name: the next arg if it does NOT start with '-' (else it's a train flag).
EXP_NAME="$CONFIG"
if [ "$#" -gt 0 ] && [[ "$1" != -* ]]; then
    EXP_NAME="$1"
    shift
fi

# Caches on the 5 TB /scratch/gpfs fileset (home is only ~50 GB on Della).
export UV_CACHE_DIR=/scratch/gpfs/TSILVER/ss1824/.cache/uv
export HF_HOME=/scratch/gpfs/TSILVER/ss1824/.cache/huggingface
export HF_HUB_CACHE=/scratch/gpfs/TSILVER/ss1824/.cache/huggingface/hub
export HF_LEROBOT_HOME=/scratch/gpfs/TSILVER/ss1824/.cache/huggingface/lerobot
export OPENPI_DATA_HOME=/scratch/gpfs/TSILVER/ss1824/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# To train without W&B instead, uncomment:
# export WANDB_MODE=offline

cd /scratch/gpfs/TSILVER/ss1824/tamp-vla/openpi

echo "=== Full FT '$CONFIG' (exp=$EXP_NAME, 1x A100-80GB, FSDP=1) on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
uv run --frozen scripts/train.py "$CONFIG" \
    --exp-name="$EXP_NAME" \
    --fsdp-devices 1 \
    --overwrite \
    "$@"
echo "=== Done at $(date) ==="
