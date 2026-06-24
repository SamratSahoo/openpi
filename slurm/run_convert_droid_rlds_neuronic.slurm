#!/bin/bash
#SBATCH --job-name=droid-rlds2lerobot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=all
#SBATCH --account=seas
#SBATCH --time=04:00:00
#SBATCH --output=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/convert_droid_%j.out
#SBATCH --error=/n/fs/tamp-vla/tamp-vla/openpi/slurm/logs/convert_droid_%j.err
#
# Neuronic equivalent of run_convert_droid_rlds.slurm (Della).
# Differences vs Della: caches live on the shared /n/fs project FS (home quota is only ~16G),
# no `module load proxy/default` (Neuronic compute nodes have direct internet), no a100
# constraint, and this job also DOWNLOADS the raw RLDS build first (CPU/IO bound -> no GPU).
#
# README step 7.1: download DROID-100 raw RLDS, then convert RLDS -> LeRobot (DROID joint schema).

set -euo pipefail

export UV_CACHE_DIR=/n/fs/tamp-vla/.cache/uv
export HF_HOME=/n/fs/tamp-vla/.cache/huggingface
export HF_HUB_CACHE=/n/fs/tamp-vla/.cache/huggingface/hub
export HF_LEROBOT_HOME=/n/fs/tamp-vla/.cache/huggingface/lerobot
export OPENPI_DATA_HOME=/n/fs/tamp-vla/.cache/openpi

# TFDS expects the builder files at <data_dir>/<builder>/<version>/. The HF repo stores them
# flat, so download straight into that nested path and point --rlds-data-dir at the data_dir root.
RLDS_DIR=$OPENPI_DATA_HOME/droid_rlds
RLDS_BUILDER_DIR=$RLDS_DIR/r2d2_faceblur/1.0.0

cd /n/fs/tamp-vla/tamp-vla/openpi

echo "=== Downloading lerobot-raw/droid_100_raw -> $RLDS_BUILDER_DIR on $(hostname) at $(date) ==="
mkdir -p "$RLDS_BUILDER_DIR"
GIT_LFS_SKIP_SMUDGE=1 uv run --frozen huggingface-cli download lerobot-raw/droid_100_raw \
    --repo-type dataset --local-dir "$RLDS_BUILDER_DIR"

echo "=== Converting r2d2_faceblur RLDS -> LeRobot (joint space) at $(date) ==="
uv run --group rlds --frozen examples/droid/convert_droid_rlds_to_lerobot.py \
    --rlds-data-dir "$RLDS_DIR" \
    --builder-name r2d2_faceblur --version 1.0.0 \
    --repo-id SamratSahoo/d100
echo "=== Done at $(date) ==="
