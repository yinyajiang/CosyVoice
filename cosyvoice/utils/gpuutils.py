import subprocess
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
  