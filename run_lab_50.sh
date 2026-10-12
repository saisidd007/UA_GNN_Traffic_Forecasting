#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

mode="${1:-fresh}"
if [[ "$mode" != fresh && "$mode" != resume ]]; then
  echo "Usage: bash run_lab_50.sh [fresh|resume]" >&2
  exit 2
fi
python_bin="${PYTHON_BIN:-.venv/bin/python}"
output_root="${OUTPUT_ROOT:-paper_results/lab_full_50_seed42}"
batch_size="${BATCH_SIZE:-64}"
threads="${THREADS:-4}"
device="${DEVICE:-auto}"
mkdir -p "$output_root"

for dataset in METR-LA PEMS-BAY; do
  output="$output_root/$dataset"
  extra=()
  if [[ "$mode" == resume && -f "$output/latest.pt" ]]; then
    extra+=(--resume)
  elif [[ -e "$output" ]]; then
    echo "Output already exists without a resumable checkpoint: $output. Use a new OUTPUT_ROOT or inspect the folder." >&2
    exit 1
  fi
  echo "Starting $dataset: seed=42, epochs=50, all windows, batch=$batch_size, threads=$threads"
  "$python_bin" -u paper_experiment.py --dataset "$dataset" \
    --output "$output" --seed 42 --epochs 50 --patience 50 \
    --hidden 64 --batch-size "$batch_size" --threads "$threads" --device "$device" \
    --train-windows 0 --mc-passes 10 --log-every 25 "${extra[@]}" \
    2>&1 | tee -a "$output_root/$dataset.log"
done
echo "Both datasets completed. Results: $output_root"
