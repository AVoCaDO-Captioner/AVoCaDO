#!/bin/bash
MODEL_PATH="path_to_AVoCaDO" # TODO
OUTPUT_DIR="$1"

mkdir -p "$OUTPUT_DIR"

python eval_scripts/DREAM-1K/generate_caption.py \
    --model_path "$MODEL_PATH" \
    --save_path "$OUTPUT_DIR/model_caption.jsonl"

bash eval_scripts/DREAM-1K/tarsier/scripts/run_evaluation_only.sh "$OUTPUT_DIR/model_caption.jsonl"

