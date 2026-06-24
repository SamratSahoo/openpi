#!/bin/bash
#SBATCH --job-name=openpi-train-extended-v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:l40:8
#SBATCH --partition=all
#SBATCH --account=seas
# neu306 reproducibly faults this run with CUDA_ERROR_ILLEGAL_ADDRESS on the first step, while the
# identical fsdp=8 step runs fine on other 8x L40 nodes -> exclude it (suspected bad GPU/node; report
# to cluster admins). Drop this line once neu306 is fixed.
#SBATCH --exclude=neu306
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/train_extended_v3_%j.out
#SBATCH --error=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/train_extended_v3_%j.err
#
# Neuronic equivalent of run_train_droid100.slurm (Della) -- FULL fine-tune of pi0.5 on the
# combined DROID-100 + corrected toys-no-collision-v2 dataset (pi05droid-full-d100+toys).
#
# Why this differs from the Della script:
#   * Full fine-tuning needs >70 GB; an L40 has only ~46 GB, so a single GPU cannot hold the
#     model + AdamW state. We grab the whole 8x L40 node and shard with FSDP across all 8
#     (--fsdp-devices 8): weights+optimizer split 8 ways (~9 GB/GPU), leaving ample room for
#     activations. The global batch (32, unchanged) is split across all 8 devices -> 4/device.
#   * Shared /n/fs caches instead of /scratch/gpfs; GPU is gpu:l40:8 (no a100 constraint);
#     no `module load proxy/default` (Neuronic compute nodes have direct internet).
#
# Prereqs: run the convert -> merge -> norm-stats jobs first (they build samratsahoo/
# droid_100_extended_v3 and its assets/ norm stats). The pi0.5-DROID base checkpoint is pulled
# from gs://openpi-assets into $OPENPI_DATA_HOME on first run.

set -euo pipefail

export UV_CACHE_DIR=/n/fs/tamp-vla/.cache/uv
export HF_HOME=/n/fs/tamp-vla/.cache/huggingface
export HF_HUB_CACHE=/n/fs/tamp-vla/.cache/huggingface/hub
export HF_LEROBOT_HOME=/n/fs/tamp-vla/.cache/huggingface/lerobot
export OPENPI_DATA_HOME=/n/fs/tamp-vla/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Weights & Biases: Neuronic compute nodes have internet, so online logging works once you have
# run `wandb login` on the login node (creds land in ~/.netrc). To run without W&B instead,
# uncomment the next line (or append `--no-wandb-enabled` to the train command below).
# export WANDB_MODE=offline

cd /n/fs/tamp-vla/tamp-vla/openpi

echo "=== Full fine-tune pi05droid-full-d100+toys (8x L40, FSDP=8) on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
uv run --frozen scripts/train.py pi05droid-full-d100+toys \
    --exp-name=droid100_extended_v3_full_ft \
    --fsdp-devices 8 \
    --overwrite
echo "=== Done at $(date) ==="
