#!/bin/bash
#SBATCH --job-name=openpi-train
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
#SBATCH --output=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/train_%x_%j.out
#SBATCH --error=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/train_%x_%j.err
#
# Generic FULL fine-tune job for any pi0.5 config on the 8x L40 / FSDP=8 setup (see
# run_train_v3_neuronic.slurm for the FSDP rationale). Pass the config name and an exp name:
#
#   sbatch --job-name=<config> slurm/run_train_neuronic.slurm <config> <exp-name>
#
# For pi05_base configs, run the matching norm-stats job first (chain with --dependency=afterok:<jid>).
# pi05_droid configs reuse the official DROID norm stats and need no norm-stats prereq.

set -euo pipefail

CONFIG="${1:?Usage: sbatch run_train_neuronic.slurm <config-name> <exp-name>}"
EXP_NAME="${2:?Usage: sbatch run_train_neuronic.slurm <config-name> <exp-name>}"

export UV_CACHE_DIR=/n/fs/tamp-vla/.cache/uv
export HF_HOME=/n/fs/tamp-vla/.cache/huggingface
export HF_HUB_CACHE=/n/fs/tamp-vla/.cache/huggingface/hub
export HF_LEROBOT_HOME=/n/fs/tamp-vla/.cache/huggingface/lerobot
export OPENPI_DATA_HOME=/n/fs/tamp-vla/.cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

cd /n/fs/tamp-vla/tamp-vla/openpi

echo "=== Full fine-tune ${CONFIG} (exp=${EXP_NAME}, 8x L40, FSDP=8) on $(hostname) at $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
uv run --frozen scripts/train.py "${CONFIG}" \
    --exp-name="${EXP_NAME}" \
    --fsdp-devices 8 \
    --overwrite
echo "=== Done at $(date) ==="
