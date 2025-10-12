import os
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import argparse
import json
from tqdm import tqdm
from pathlib import Path

VIDEO_MAX_PIXELS = 401408  # 512*28*28
VIDEO_TOTAL_PIXELS = 20070400  # 512*28*28*50
USE_AUDIO_IN_VIDEO = False
os.environ['VIDEO_MAX_PIXELS'] = str(VIDEO_TOTAL_PIXELS)
script_dir = Path(__file__).resolve().parent
example_path = script_dir / "dream_example.jsonl"
video_dir = "" # TODO

parser = argparse.ArgumentParser(description="Evaluate a model and save results.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint.")
parser.add_argument("--save_path", type=str, required=True, help="Path to save the evaluation results.")
args = parser.parse_args()

model_path = args.model_path
fout_path = args.save_path

f_example = open(example_path, 'r', encoding='utf-8')
fout = open(fout_path, 'w', encoding='utf-8')

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model.disable_talker()
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

def chat(data):
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": data["video_path"],
                    "max_pixels": VIDEO_MAX_PIXELS,
                },
                {
                    "type": "text",
                    "text": data["question"]
                },
            ],
        },
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)

    text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, do_sample=False, thinker_max_new_tokens=2048)

    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    model_generation = text.split("\nassistant\n")[-1]

    return model_generation


for idx, line in tqdm(enumerate(f_example, start=1)):
    data = json.loads(line)
    video_path = os.path.join(video_dir, data["messages"][0]["content"][0]["video"]["video_file"])
    question = "Imagine the video from these frames and describe it in detail."

    temp_data = {
        "video_path": video_path,
        "question": question,
        }
    with torch.inference_mode():
        response = chat(temp_data)

        out_data = data
        data["messages"][0]["content"][1]["text"] = question
        out_data["messages"][1]["content"][0]["text"] = response
        fout.write(json.dumps(out_data, ensure_ascii=False) + '\n')
        fout.flush()
