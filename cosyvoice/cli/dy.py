import torch
import gc
import contextlib
import time
from cosyvoice.cli.vllm_cosvoice import AutoCosyVoice
from cosyvoice.utils.gpuutils import get_gpu_total_memory_mb, get_gpu_memory_mb
import os


def _recommend_trt_concurrent_fast(model):
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
    return trt_concurrent
    

def _recommend_trt_concurrent_calc(model):
    concurrent = 0
    with contextlib.suppress(Exception):
        while True:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            time.sleep(2)
            AutoCosyVoice(
                model_dir=model,
                fp16=False,
                load_vllm=False,
                load_trt=True,
                trt_concurrent=concurrent+1
            )
            concurrent += 1
    return concurrent


def recommend_trt_concurrent(model, fast=True):
    if fast:
        return _recommend_trt_concurrent_fast(model)
    else:
        return _recommend_trt_concurrent_calc(model)


def default_zero_shot_prompt_wav(model_dir):
    return os.path.abspath(os.path.join(model_dir, '../../asset/zero_shot_prompt.wav'))
