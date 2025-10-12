import os
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import argparse
import json
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
import traceback
import random
import glob

VIDEO_MAX_PIXELS = 401408  # 512*28*28
VIDEO_TOTAL_PIXELS = 20070400  # 512*28*28*50
USE_AUDIO_IN_VIDEO = True
video_base_dir = "path_to_WorldSense_videos"
os.environ['VIDEO_MAX_PIXELS'] = str(VIDEO_TOTAL_PIXELS)

def chat(file_path, prompt, model, processor, model_path, max_new_tokens=2048):
    
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
                    "max_frames": 256
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

    text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, do_sample=False, thinker_max_new_tokens=max_new_tokens)
    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    model_generation = text.split("\nassistant\n")[-1]
    
    return model_generation

def worker_proc(rank, gpu_id, model_path, video_paths, prompt, out_path):
    device_map = {"": f"cuda:{gpu_id}"}

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

    fout = open(out_path, "w", encoding="utf-8")

    for video_path in tqdm(video_paths, desc=f"Worker-{rank}[GPU-{gpu_id}]"):
        try:
            model_generation = chat(video_path, prompt, model, processor, model_path)

            video_id = os.path.basename(video_path).split(".mp4")[0]
            out_data = {
                "video_id": video_id,
                "caption": model_generation,
            }

            fout.write(json.dumps(out_data, ensure_ascii=False) + "\n")
            fout.flush()
        except Exception as e:
            print(f"[Worker-{rank}] Error on {video_path}: {e}")
            traceback.print_exc()

    fout.close()
    print(f"[Worker-{rank}] Done, wrote results to {out_path}")

def run_multi_gpu(model_path, video_paths, prompt_list, final_out_path, num_gpus=8):
    chunk_size = len(video_paths) // num_gpus + 1
    chunks = [video_paths[i:i+chunk_size] for i in range(0, len(video_paths), chunk_size)]

    processes = []
    tmp_files = []

    for rank, chunk in enumerate(chunks):
        gpu_id = rank % num_gpus
        tmp_out = final_out_path.replace(".jsonl", f".part{rank}.jsonl")
        tmp_files.append(tmp_out)
        prompt = random.choice(prompt_list)

        p = mp.Process(
            target=worker_proc,
            args=(rank, gpu_id, model_path, chunk, prompt, tmp_out)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    with open(final_out_path, "w", encoding="utf-8") as fout:
        for tmp in tmp_files:
            with open(tmp, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
            os.remove(tmp)

    print(f"All results merged into {final_out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model and save results.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--fout_path", type=str, required=True, help="Path to the output caption file")
    args = parser.parse_args()
    mp.set_start_method("spawn", force=True) 

    video_paths = glob.glob(os.path.join(video_base_dir, "**", "*.mp4"), recursive=True)

    prompt_list = [
    "Provide a comprehensive description of all the content in the video, leaving out no details. Be sure to include as much of the audio information as possible, and ensure that your descriptions of the audio and video are closely aligned.",
    "Thoroughly describe everything in the video, capturing every detail. Include as much information from the audio as possible, and ensure that the descriptions of both audio and video are well-coordinated.",
    "Please describe all the information in the video without sparing every detail in it. As you describe, you should also describe as much of the information in the audio as possible, and pay attention to the synchronization between the audio and video descriptions.",
    "Offer a detailed description of the video, making sure to include every detail. Also, incorporate as much information from the audio as you can, and ensure that your descriptions of the audio and video are in sync.",
    "Describe every aspect of the video in full detail, covering all the information it contains. Additionally, include as much of the audio content as you can, and make sure your descriptions of the audio and video are synchronized.",
    "Please provide a thorough description of all the content in the video, including every detail. As you describe, ensure that you also cover as much information from the audio as possible, and be mindful of the synchronization between the audio and video as you do so.",
    "Give a detailed account of everything in the video, capturing all the specifics. While doing so, also include as much information from the audio as possible, ensuring that the descriptions of audio and video are well-synchronized."
    ]

    run_multi_gpu(args.model_path, video_paths, prompt_list, args.fout_path, num_gpus=8)
