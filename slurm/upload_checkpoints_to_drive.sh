#!/bin/bash
# One-off: upload the 3 final (step-19999) checkpoints to their Google Drive folders via the
# configured rclone `drive:` remote. Each checkpoint lands in a `19999/` subfolder of the target.
set -uo pipefail

CKPT_BASE=/n/fs/tamp-vla/tamp-vla/openpi/checkpoints

declare -a JOBS=(
  "$CKPT_BASE/pi05_droid100_extended_v3_lerobot/droid100_extended_v3_full_ft/19999|17JIDRFv6sJrmRdT4ClBWlakSWfpz-reR"
  "$CKPT_BASE/pi05_droid100_from_base_lerobot/droid100_from_base_full_ft/19999|1-SSEfu9yt3kOnf42E1FmN4DFkWiDbMbL"
  "$CKPT_BASE/pi05_droid100_extended_v3_from_base_lerobot/droid100_extended_v3_from_base_full_ft/19999|1agytl9wnNVnyrVdJiwK41EdQ4HbXuHNU"
)

for job in "${JOBS[@]}"; do
  src="${job%%|*}"
  fid="${job##*|}"
  echo "=== [$(date)] uploading $src -> drive folder $fid (as 19999/) ==="
  rclone copy "$src" "drive:19999" \
    --drive-root-folder-id "$fid" \
    --transfers 4 --checkers 8 --drive-chunk-size 128M \
    --stats 30s --stats-one-line --log-level INFO
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!! [$(date)] FAILED ($rc) for $src -> $fid"
    exit $rc
  fi
  echo "=== [$(date)] verifying $src -> $fid ==="
  rclone check "$src" "drive:19999" --drive-root-folder-id "$fid" --one-way
  echo "=== [$(date)] done $src ==="
done
echo "=== [$(date)] ALL UPLOADS COMPLETE ==="
