raw_file="$1"

output_file="${raw_file/caption.jsonl/results.jsonl}"

python eval_scripts/VDC/score_sglang_multi-threads.py \
    --raw_file $raw_file \
    --output_file $output_file