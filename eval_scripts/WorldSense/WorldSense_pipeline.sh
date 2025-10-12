#!/bin/bash
MODEL_PATHS=(
    "path_to_AVoCaDO"
)
RESULTS_DIR="$1"

ORIGINAL_FILE="eval_scripts/WorldSense/worldsense_qa.json"
MERGED_FILE="$RESULTS_DIR/captioned_results.json"
if [ ! -f "$MERGED_FILE" ]; then
    echo "MERGED_FILE not found. Creating from ORIGINAL_FILE..."
    cp "$ORIGINAL_FILE" "$MERGED_FILE"
fi
CAPTION_FILES_TO_MERGE=()
CAPTION_KEYS=()

# Step 1: caption geneartion
for model_path in "${MODEL_PATHS[@]}"; do
    CLEAN_PATH="${model_path%/}"
    model_name=$(basename "$CLEAN_PATH")
    
    caption_file="$RESULTS_DIR/${model_name}_caption.jsonl"
    echo "Output caption file will be: $caption_file"

    python eval_scripts/WorldSense/generate_caption.py \
        --model_path "$model_path" \
        --fout_path "$caption_file" 
    
    if [ -f "$caption_file" ]; then
        CAPTION_FILES_TO_MERGE+=("$caption_file")
        CAPTION_KEYS+=("${model_name}_caption")
    else
        echo "Error: Caption file $caption_file not generated for model $model_path."
        exit 1
    fi
done

# Step 2: merge generated caption files
echo "Merging all generated caption files..."
python eval_scripts/WorldSense/merge_captions.py \
    --original_file "$MERGED_FILE" \
    --caption_files "${CAPTION_FILES_TO_MERGE[@]}" \
    --merged_file "$MERGED_FILE"

# Step 3: evaluation
python eval_scripts/WorldSense/evaluation.py \
    --merged_file "$MERGED_FILE" \
    --caption_keys "${CAPTION_KEYS[@]}"

# Step 4: analysis and save evaluation results
for caption_key in "${CAPTION_KEYS[@]}"; do
    echo "Running analysis for caption key: $caption_key"
    
    result_file="$RESULTS_DIR/${caption_key}_result.jsonl"
    answer_key="${caption_key//_caption/_resp}"
    
    if [ -f "$result_file" ]; then
        python eval_scripts/WorldSense/analysis.py --result_file_path "$result_file" --answer_key "$answer_key"
    else
        echo "Warning: Result file '$result_file' not found for analysis."
    fi
done
