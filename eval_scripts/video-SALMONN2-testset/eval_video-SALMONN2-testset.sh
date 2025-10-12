#!/bin/bash
MODEL_PATH="path_to_AVoCaDO" # TODO
OUTPUT_DIR="$1"

mkdir -p "$OUTPUT_DIR"

python eval_scripts/video-SALMONN2-testset/generate_caption.py \
    --model_path "$MODEL_PATH" \
    --save_path "$OUTPUT_DIR/model_caption.json"

python eval_scripts/video-SALMONN2-testset/evaluation.py "$OUTPUT_DIR/model_caption.json"
