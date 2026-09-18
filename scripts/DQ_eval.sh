#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 <aitod_v2_path> <checkpoint> [output_dir]" >&2
  exit 1
fi

dataset_path="$1"
checkpoint="$2"
output_dir="${3:-logs/DQ_eval}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

python "$repo_root/main_aitod.py" \
  --output_dir "$output_dir" \
  -c "$repo_root/config/DQ_5scale.py" \
  --dataset_file aitod_v2 \
  --coco_path "$dataset_path" \
  --eval \
  --resume "$checkpoint" \
  --visualize \
  --vis_score_thresh 0.3 \
  --options dn_number=100 embed_init_tgt=False use_ema=False \
  dn_box_noise_scale=1.0
