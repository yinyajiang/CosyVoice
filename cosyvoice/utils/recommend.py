import torch
import subprocess
import gc
import time
import os


def get_gpu_memory_mb():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        return int(result.stdout.strip().split('\n')[0])
    return None


def get_gpu_total_memory_mb():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        return int(result.stdout.strip().split('\n')[0])
    return None


def recommend_trt_concurrent(model):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)

    if 'Fun-CosyVoice3-0.5B' in model:
        base = 8400
        increment = 2500
    else:
        raise ValueError(f'Unsupported model: {model}')
    print(f"Model: {model}, base: {base}MB, increment: {increment}MB")
    total_memory = get_gpu_total_memory_mb()
    print(f"GPU total memory: {total_memory} MB")
    if total_memory is None:
        raise ValueError('Failed to get GPU total memory')
    current_memory = get_gpu_memory_mb()
    print(f"GPU current memory: {current_memory} MB")
    if current_memory is None:
        raise ValueError('Failed to get GPU current memory')
    free_memory = total_memory - current_memory
    print(f"GPU free memory: {free_memory} MB")
    if free_memory < base:
        raise ValueError(f'Not enough free memory: {free_memory} MB < {base} MB')
    trt_concurrent = (free_memory - base) // increment + 1
    print(f"Recommended trt_concurrent: {trt_concurrent}")
    return trt_concurrent
    

def default_zero_shot_prompt_wav(model_dir):
    return os.path.abspath(os.path.join(model_dir, '../../asset/zero_shot_prompt.wav'))