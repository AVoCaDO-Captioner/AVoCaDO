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
USE_AUDIO_IN_VIDEO = True
os.environ['VIDEO_MAX_PIXELS'] = str(VIDEO_TOTAL_PIXELS)
video_dir = "" # TODO


parser = argparse.ArgumentParser(description="Evaluate a model and save results.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint.")
parser.add_argument("--save_path", type=str, required=True, help="Path to save the evaluation results.")
args = parser.parse_args()

model_path = args.model_path
fout_path = args.save_path

fout = open(fout_path, 'w', encoding='utf-8')

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model.disable_talker()
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)


def chat(file_path, prompt):
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
                    "video": file_path,
                    "max_pixels": VIDEO_MAX_PIXELS,
                },
                {
                    "type": "text",
                    "text": prompt
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

for idx, video_id in tqdm(enumerate(os.listdir(video_dir), start=1)):
    video_path = os.path.join(video_dir, video_id)
    prompt = ("You are given a short video with both audio and visual content. Write a detailed and coherent paragraph that naturally integrates all modalities. "
    "Your description should include: (1) the primary scene and background setting; (2) key characters or objects and their actions or interactions; "
    "(3) significant audio cues such as voices, background music, sound effects, and their emotional tone; "
    "(4) any on-screen text (OCR) and its role in the video context; and (5) the overall theme or purpose of the video. "
    "Ensure the output is a fluent and objective paragraph, not a bullet-point list, and captures the video's content in a human-like, narrative style.")

    model_generation = chat(video_path, prompt)
    out_data = {
        "video_id": video_id.replace(".mp4", ""),
        "caption": model_generation, 
    }
    fout.write(json.dumps(out_data, ensure_ascii=False)+'\n')
    fout.flush()