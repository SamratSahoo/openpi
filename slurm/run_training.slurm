#!/bin/bash
#SBATCH --job-name=openpi-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:l40:8
#SBATCH --partition=all
#SBATCH --account=seas
# neu306 reproducibly faults fsdp=8 runs on the first step (CUDA_ERROR_ILLEGAL_ADDRESS) while other
# 8x L40 nodes are fine -- exclude it. Drop this once the node is fixed.
#SBATCH --exclude=neu306
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/train_%j.out
#SBATCH --error=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/train_%j.err
#
# Full fine-tune ANY openpi TrainConfig on Neuronic. Takes the whole 8x L40 node and shards the
# model + AdamW state with FSDP across all 8 GPUs (--fsdp-devices 8): an L40 is ~46 GB but full FT
# of pi0.5 needs >70 GB, so a single GPU can't hold it; sharded it's ~9 GB/GPU. Global batch stays
# as configured (32 -> 4/device). If only 4 GPUs are free, edit --gres to gpu:l40:4 and pass
# "--fsdp-devices 4" (and, on OOM, "--batch-size 16").
#
#   Usage:  sbatch run_training.slurm <config_name> [exp_name] [extra train.py args]
#   e.g.:   sbatch run_training.slurm pi05base-full-d100+toys100sim-stream
#           sbatch run_training.slurm pi05base-toys20sim my_run --batch-size 16
#
# exp_name defaults to the config name. Run compute_norm_stats.slurm for the same config first.
# W&B logs online once you've run `wandb login` on the login node.

set -euo pipefail

CONFIG="${1:?usage: sbatch run_training.slurm <config_name> [exp_name] [extra args]}"
shift
# Optional positional exp_name: the next arg if it does NOT start with '-' (else it's a train flag).
EXP_NAME="$CONFIG"
if [ "$#" -gt 0 ] && [[ "$1" != -* ]]; then
    EXP_NAME="$1"
    shift
fi

# Caches on the 1 TB project FS (home is only ~16 GB on Neuronic).
export UV_CACHE_DIR=/n/fs/tamp-vla/.cache/uv
export HF_HOME=/n/fs/tamp-vla/.cache/huggingface
export HF_HUB_CACHE=/n/fs/tamp-vla/.cache/huggingface/hub
export HF_LEROBOT_HOME=/n/fs/tamp-vla/.cache/huggingface/lerobot
export OPENPI_DATA_HOME=/n/fs/tamp-vla/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# To train without W&B instead, uncomment:
# export WANDB_MODE=offline

cd /n/fs/tamp-vla/tamp-vla/openpi

echo "=== Full FT '$CONFIG' (exp=$EXP_NAME, 8x L40, FSDP=8) on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
uv run --frozen scripts/train.py "$CONFIG" \
    --exp-name="$EXP_NAME" \
    --fsdp-devices 8 \
    --overwrite \
    "$@"
echo "=== Done at $(date) ==="
