import pandas as pd
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the evaluation results.")
    parser.add_argument("--result_file_path", type=str, required=True, help="Path to the result file (.jsonl).")
    parser.add_argument("--answer_key", type=str, required=True, help="The key for the model's response in the result file.")
    
    args = parser.parse_args()

    data = pd.read_json(args.result_file_path, lines=True)

    acc = (data['answer'].str.upper() == data[args.answer_key].str.upper()).mean()
    print(f"Accuracy for {args.answer_key} is: {acc:.2%}")

    with open(f"{os.path.dirname(args.result_file_path)}/{args.answer_key}.log", "w", encoding='utf-8') as fout:
        fout.write(f"Accuracy for {args.answer_key} is: {acc:.2%}")