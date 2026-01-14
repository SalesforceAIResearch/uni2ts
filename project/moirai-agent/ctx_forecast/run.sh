#!/bin/bash
set -e

# --------------------------------------------------
# Positional args with defaults
# --------------------------------------------------
TASK_FILE=${1:-"gift_ctx_images.parquet"}
CONFIG_FILE=${2:-"src/ctx_forecast/config.py"}
OUTPUT_DIR=${3:-"results/moirai_agent"}
INPUT_MODE=${4:-"text+image"}
JOBS=${5:-$(nproc 2>/dev/null || echo 8)}
NUM_PARTS=${6:-245}

# --------------------------------------------------
# Run all parts in parallel
# --------------------------------------------------
seq 0 $((NUM_PARTS - 1)) | parallel -j "$JOBS" \
  python -m src.ctx_forecast.tsf_agent \
    --config_file "$CONFIG_FILE" \
    --config_name CONFIG \
    --input_mode "$INPUT_MODE" \
    --task_file "$TASK_FILE" \
    --part_idx {} \
    --num_parts "$NUM_PARTS" \
    --output_dir "$OUTPUT_DIR"

echo "Finish all parts"
echo "Calculating metrics"

# --------------------------------------------------
# Gather metrics
# --------------------------------------------------
python -m src.ctx_forecast.metrics_gather \
  --results_dir "${OUTPUT_DIR}/${INPUT_MODE}/results"

echo "Done! Results are available at ${OUTPUT_DIR}/${INPUT_MODE}"

