#!/bin/bash
MODEL_PATH="path_to_AVoCaDO" # TODO
OUTPUT_DIR="$1"

mkdir -p "$OUTPUT_DIR"

python eval_scripts/UGC-VideoCap/generate_caption.py \
    --model_path "$MODEL_PATH" \
    --save_path "$OUTPUT_DIR/model_caption.jsonl"

python eval_scripts/UGC-VideoCap/evaluation.py "${OUTPUT_DIR}/model_caption.jsonl" "${OUTPUT_DIR}/eval_results.json"
