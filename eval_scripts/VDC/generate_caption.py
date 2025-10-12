import os
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import argparse
import json
from tqdm import tqdm
import multiprocessing as mp
import re
import traceback
import logging
import sys
import time
from multiprocessing import Value, Lock
from torch.utils.data import Dataset, DataLoader
from functools import partial
import glob
import gc


VIDEO_MAX_PIXELS = 401408  # 512*28*28
VIDEO_TOTAL_PIXELS = 20070400  # 512*28*28*50
USE_AUDIO_IN_VIDEO = False
video_dir = "path_to_Video-Detailed-Caption_videos"
os.environ['VIDEO_MAX_PIXELS'] = str(VIDEO_TOTAL_PIXELS)

def setup_logger(rank, log_dir):
    logger = logging.getLogger(f"worker_{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        log_file = os.path.join(log_dir, f"eval_rank_{rank}.log")
        file_handler = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - (%(processName)s) - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

class VideoDataset(Dataset):
    def __init__(self, video_paths, prompts):
        self.video_paths = video_paths
        self.prompts = prompts

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        prompt = self.prompts[idx]
        
        conversation = [
            {
                "role": "system", 
                "content": [
                    {
                        "type": "text", 
                        "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                        }
                    ]
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "video", 
                        "video": video_path, 
                        "max_pixels": VIDEO_MAX_PIXELS, 
                        "max_frames": 256
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    },
                    ]
            },
        ]
        return conversation, video_path

def collate_fn(batch, processor):
    conversations = [item[0] for item in batch]
    video_paths = [item[1] for item in batch]
    
    texts = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    audios, images, videos_pre = process_mm_info(conversations, use_audio_in_video=USE_AUDIO_IN_VIDEO)

    if videos_pre:
        max_frames = max(video.shape[0] for video in videos_pre)
        padded_videos = []
        for video in videos_pre:
            current_frames, C, H, W = video.shape
            padding_needed = max_frames - current_frames
            if padding_needed > 0:
                padding_tensor = torch.zeros((padding_needed, C, H, W), dtype=video.dtype)
                padded_video = torch.cat([video, padding_tensor], dim=0)
                padded_videos.append(padded_video)
            else:
                padded_videos.append(video)
        videos = padded_videos
    else:
        videos = videos_pre

    inputs = processor(text=texts, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    
    return inputs, video_paths

def generate_captions(batch_inputs, model, processor, model_path):
    text_ids = model.generate(**batch_inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, do_sample=False, thinker_max_new_tokens=2048)

    decoded_texts = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    results = []
    for text in decoded_texts:
        model_generation = text.split("\nassistant\n")[-1]
        results.append(model_generation)
    return results

def worker_proc(rank, gpu_id, model_path, all_video_paths, all_prompts, out_path, counter, lock, log_dir):
    logger = setup_logger(rank, log_dir)
    logger.info(f"Worker-{rank} started on GPU-{gpu_id}, process ID: {os.getpid()}")

    device_map = {"": f"cuda:{gpu_id}"}

    try:
        logger.info("Loading model...")
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device_map, attn_implementation="flash_attention_2")
        model.disable_talker()
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return

    all_prompts = list(all_prompts)

    num_gpus = torch.cuda.device_count()
    video_paths_subset = all_video_paths[rank::num_gpus]
    prompts_subset = all_prompts[rank::num_gpus]
    dataset = VideoDataset(video_paths_subset, prompts_subset)
    collate_with_processor = partial(collate_fn, processor=processor)

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,
        collate_fn=collate_with_processor,
        pin_memory=True,
    )

    fout = open(out_path, "w", encoding="utf-8")
    
    for batch_inputs, batch_video_paths in dataloader:
        logger.info(f"Processing batch of {len(batch_video_paths)} videos. First video: {os.path.basename(batch_video_paths[0])}")
        model_generations = None
        try:
            batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
            model_generations = generate_captions(batch_inputs, model, processor, model_path)

            for video_path, model_generation in zip(batch_video_paths, model_generations):
                video_id = os.path.basename(video_path).split(".mp4")[0]
                if model_path.endswith("HumanOmniV2"):
                    context = extract_context(model_generation)
                    thinking = extract_think(model_generation)
                    answer = extract_answer(model_generation)
                    out_data = {"video_id": video_id, "context": context, "thinking": thinking, "caption": answer}
                else:
                    out_data = {"video_id": video_id, "caption": model_generation}
                fout.write(json.dumps(out_data, ensure_ascii=False) + "\n")

            fout.flush()
            with lock:
                counter.value += len(batch_video_paths)
        except Exception as e:
            logger.error(f"CAUGHT PYTHON EXCEPTION on batch starting with {os.path.basename(batch_video_paths[0])}: {e}", exc_info=True)
        finally:
            del batch_inputs
            del model_generations
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    fout.close()
    logger.info(f"Worker finished processing all its data.")

def run_multi_gpu(model_path, video_paths, prompts, final_out_path, num_gpus=8):
    log_dir = os.path.join(os.path.dirname(final_out_path), "eval_logs")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Log files will be saved in: {log_dir}")

    processes = []
    tmp_files = []
    counter = Value("i", 0)
    lock = Lock()
    total = len(video_paths)

    for rank in range(num_gpus):
        gpu_id = rank % num_gpus
        tmp_out = final_out_path.replace(".jsonl", f".part{rank}.jsonl")
        tmp_files.append(tmp_out)
        
        p = mp.Process(
            target=worker_proc,
            args=(rank, gpu_id, model_path, video_paths, prompts, tmp_out, counter, lock, log_dir),
            name=f"Worker-{rank}"
        )
        p.start()
        processes.append(p)
    
    with tqdm(total=total, desc="Processing videos", ncols=100) as pbar:
        last_count = 0
        while True:
            with lock:
                current_count = counter.value
            if current_count > last_count:
                pbar.update(current_count - last_count)
                last_count = current_count
            if current_count >= total:
                break
            time.sleep(1)

    for p in processes:
        p.join()

    print("Merging results...")
    with open(final_out_path, "a", encoding="utf-8") as fout:
        for tmp in tmp_files:
            if os.path.exists(tmp):
                with open(tmp, "r", encoding="utf-8") as fin:
                    fout.write(fin.read())
                os.remove(tmp)
    print(f"New results have been appended to {final_out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model and save results.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--fout_path", type=str, required=True, help="Path to the output caption file.")
    args = parser.parse_args()
    
    prompt = "Describe every aspect of the video in full detail, covering all the information it contains."
    
    mp.set_start_method("spawn", force=True) 
    
    completed_video_ids = set()
    if os.path.exists(args.fout_path):
        print(f"Found existing output file at: {args.fout_path}. Attempting to resume.")
        with open(args.fout_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'video_id' in data:
                        completed_video_ids.add(data['video_id'])
                except json.JSONDecodeError:
                    print(f"Warning: Skipping corrupted line in existing output file: {line.strip()}")
        print(f"Loaded {len(completed_video_ids)} completed video IDs from the existing file.")

    if not os.path.isdir(video_dir):
        print(f"Error: Provided video directory does not exist: {video_dir}")
        sys.exit(1)
        
    print(f"Scanning for .mp4 files in {video_dir}...")
    all_video_paths = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
    
    if not all_video_paths:
        print(f"Error: No .mp4 files found in {video_dir}")
        sys.exit(1)
        
    print(f"Found {len(all_video_paths)} videos.")

    videos_to_process = []
    for path in all_video_paths:
        video_id = os.path.basename(path).replace(".mp4", "")
        if video_id not in completed_video_ids:
            videos_to_process.append(path)

    if not videos_to_process:
        print("All videos have already been processed. Nothing to do. Exiting.")
        sys.exit(0)

    print(f"Identified {len(videos_to_process)} videos remaining to be processed for this run.\n")

    prompts = [prompt] * len(videos_to_process)

    run_multi_gpu(args.model_path, videos_to_process, prompts, args.fout_path, num_gpus=8)