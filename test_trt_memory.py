#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试不同 trt_concurrent 值下的显存占用
"""
import torch
import gc
from cosyvoice.cli.vllm_cosvoice import AutoCosyVoice
from cosyvoice.cli.dy import recommend_trt_concurrent
from cosyvoice.utils.gpuutils import get_gpu_memory_mb, get_gpu_total_memory_mb
import argparse
import os


def find_trt_base(trt_concurrent_values, model):
    # 清空显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    baseline_memory = get_gpu_memory_mb()
    total_memory = get_gpu_total_memory_mb()
    print(f"\n基线显存占用: {baseline_memory} MB")
    if total_memory:
        print(f"GPU总显存: {total_memory} MB")
        print(f"基线剩余显存: {total_memory - baseline_memory} MB")
    
    results = []
    
    for trt_concurrent in trt_concurrent_values:
        print(f"\n测试 trt_concurrent = {trt_concurrent}")
        print("-" * 60)
        
        # 清空显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        memory_before = get_gpu_memory_mb()
        print(f"加载模型前显存: {memory_before} MB")
        
        try:
            # 加载模型
            cosyvoice = AutoCosyVoice(
                model_dir=model,
                fp16=False,
                load_vllm=False,
                load_trt=True,
                trt_concurrent=trt_concurrent
            )
            
            # 等待一下让显存稳定
            import time
            time.sleep(2)
            
            memory_after = get_gpu_memory_mb()
            model_memory = memory_after - memory_before
            free_memory = total_memory - memory_after if total_memory else None
            print(f"加载模型后显存: {memory_after} MB")
            print(f"模型占用显存: {model_memory} MB")
            if free_memory is not None:
                print(f"剩余显存: {free_memory} MB")
            
            results.append({
                'trt_concurrent': trt_concurrent,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'model_memory': model_memory,
                'free_memory': free_memory
            })
            
            # 清理
            del cosyvoice
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"错误: {e}")
            results.append({
                'trt_concurrent': trt_concurrent,
                'error': str(e)
            })
    
    # 打印结果总结
    print("\n" + "=" * 80)
    print("结果总结")
    print("=" * 80)
    print(f"{'trt_concurrent':<15} {'显存占用(MB)':<15} {'剩余显存(MB)':<15} {'增量(MB)':<15}")
    print("-" * 80)
    
    prev_memory = None
    for i, result in enumerate(results):
        if 'error' in result:
            print(f"{result['trt_concurrent']:<15} {'ERROR':<15} {'-':<15} {'-':<15}")
            continue
            
        trt_concurrent = result['trt_concurrent']
        model_memory = result['model_memory']
        free_memory = result.get('free_memory', '-')
        free_memory_str = f"{free_memory}" if free_memory != '-' else '-'
        
        if prev_memory is not None:
            increment = model_memory - prev_memory
            print(f"{trt_concurrent:<15} {model_memory:<15} {free_memory_str:<15} {increment:<15}")
        else:
            print(f"{trt_concurrent:<15} {model_memory:<15} {free_memory_str:<15} {'-':<15}")
        
        prev_memory = model_memory
    
    # 计算平均增量
    valid_results = [r for r in results if 'error' not in r]
    if len(valid_results) > 1:
        increments = []
        for i in range(1, len(valid_results)):
            increment = valid_results[i]['model_memory'] - valid_results[i-1]['model_memory']
            increments.append(increment)
        
        if increments:
            avg_increment = sum(increments) / len(increments)
            print("-" * 80)
            print(f"\n平均每个 trt_concurrent 增量: {avg_increment:.2f} MB")
            print(f"每个 execution context 约占用: {avg_increment:.2f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='pretrained_models/Fun-CosyVoice3-0.5B')
    parser.add_argument('--find-base', type=int, default=0)
    args = parser.parse_args()

    if args.find_base:
        print(f"find base for {args.model} ...")
        find_trt_base(range(1, args.find_base+1), args.model)
        os.exit(0)

    r = recommend_trt_concurrent(args.model, True)
    print(f"fast recommended trt concurrent: {r}")
    print("\n\ntest trt concurrent ...")
    
    concurrent = recommend_trt_concurrent(args.model, False)

    print("\n\n#########\nresult:")
    if concurrent == r:
        print(f"   recommended trt concurrent: {r}")
    else:
        print(f"   max concurrent: {concurrent}, but recommended trt concurrent: {r}")
    print("#########\n")

