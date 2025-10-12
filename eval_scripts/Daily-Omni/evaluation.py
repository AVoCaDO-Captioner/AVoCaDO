### 
# using a llm to answer questions regarding to the video with the specific caption
###
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS']=''
LOCATION = "global"
user_info_path = ''
user_info = json.load(open(user_info_path))
PROJECT_ID = user_info['project_id']
MODEL = "gemini-2.5-pro"

import sys
import time
import json
import traceback
import multiprocessing
import random
import numpy as np
import argparse
from google import genai
from google.genai import types
from IPython.display import HTML, Image, Markdown, display
from google import genai
from google.genai.types import (
    FunctionDeclaration,
    GenerateContentConfig,
    GoogleSearch,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    SafetySetting,
    ThinkingConfig,
    Tool,
    ToolCodeExecution,
)
import subprocess

safety_settings = [
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.OFF),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.OFF),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.OFF),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.OFF)
]

CONFIG = types.GenerateContentConfig(
    temperature=0,
    top_p=0.001,
    thinking_config=types.ThinkingConfig(
      include_thoughts=True,
      thinking_budget=512
    ),
    safety_settings=safety_settings,
    seed=SEED,
    system_instruction='''
    You are a precise QA assistant. Your task is to answer multiple-choice questions based ONLY on the video caption provided. 
    Do not use any outside knowledge or assumptions—your answer must strictly reflect information from the caption. 
    Always output only the capital letter corresponding to your choice (e.g., A, B, C, D). 
    If the caption does not provide enough information to answer the question, output "N/A" instead.
    '''
)
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

SEED = 42
set_seed(SEED)

def caption2json(json_path, caption_path):

    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    model = os.path.basename(caption_path).split("_")[0]

    captions = {}
    with open(caption_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            for vid, cap in item.items():
                captions[vid] = cap
    
    for entry in json_data:
        vid = entry.get("video_id")
        if vid in captions:
            entry[f"{model}_caption"] = captions[vid]
    
    with open(f"{model}_merge_data.json", 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"merged successfully, the output file is {model}_merge_data.json")


def generate(prompt):
    contents = [prompt]

    answer, thinking = None, None
    max_retries = 10

    for i in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=CONFIG
                )

            answer_parts, thought_parts = [], []
            for part in response.candidates[0].content.parts:
                if not getattr(part, "text", None):
                    continue
                if getattr(part, "thought", False):
                    thought_parts.append(part.text)
                else:
                    answer_parts.append(part.text)
            answer = "\n".join(answer_parts).strip()
            thinking = "\n".join(thought_parts).strip()
            if answer:
                break
            else:
                print(f"[WARN] Attempt {i+1}: empty answer, retrying ... ")
                time.sleep(3)
        except Exception as e:
            print(f"[ERROR] Attempt {i+1} failed: {e}")
            traceback.print_exc()
            time.sleep(3)
    if not answer:
        return None, None
    print(answer)
    return answer, thinking

def worker(task):
    vid, video_duration, question, choices, answer, caption_key, answer_key, caption = task
    choices_text = "\n".join([f"{c}" for c in choices])
    prompt_filled = f'''
Here is the video caption:
"{caption}"

Question: {question}
Choices:
    {choices_text}'''
    try:
        resp, _ = generate(prompt_filled)
        return {
            "video_id": vid,
            "video_duration": video_duration,
            "question": question,
            "choices": choices,
            "answer": answer,
            caption_key: caption,
            answer_key: resp
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "video_id": vid,
            "video_duration": video_duration,
            "question": question,
            "choices": choices,
            "answer": answer,
            caption_key: caption,
            answer_key: None
        }

def run_multiprocess_tasks(tasks, num_processes=None, fout_path=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(worker, tasks)

    if fout_path:
        with open(fout_path, "w", encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                f.flush()
    return results

def eval_dailyomni_caption_qas(file_path, caption_keys=["omni_caption"]):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_results = []
    for caption_key in caption_keys:
        answer_key = caption_key.replace("_caption", "_resp")
        fout_path = f"{os.path.dirname(file_path)}/{caption_key}_result.jsonl"
        
        tasks = []
        for video_info in data:
            vid = video_info["video_id"]
            video_duration = video_info["video_duration"]
            caption = video_info[caption_key]
            for q in video_info["questions"]:
                task_item = (
                    vid,
                    video_duration,
                    q["Question"],
                    q["Choice"],
                    q["Answer"],
                    caption_key,
                    answer_key,
                    caption
                )
                tasks.append(task_item)

        results = run_multiprocess_tasks(tasks, num_processes=20, fout_path=fout_path)
        all_results.extend(results)

    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate captions using Gemini.")
    parser.add_argument("--merged_file", type=str, required=True, help="Path to the merged caption file.")
    parser.add_argument(
        "--caption_keys", 
        type=str, 
        nargs='+',
        required=True, 
        help="A list of caption keys to evaluate"
    )
    args = parser.parse_args()

    eval_dailyomni_caption_qas(args.merged_file, caption_keys=args.caption_keys)