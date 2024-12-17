import torch
from torch.amp import autocast
import time


def benchmark_model(model, tokenizer, texts, device='cpu', seq_len=512):
    """Benchmark model for inference time and memory usage."""
    inputs = tokenizer(texts, return_tensors="pt", max_length=seq_len, padding=True, truncation=True).to(device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    else:
        print("CUDA is not available. Skipping GPU memory reset.")
    times = []

    # Warmup
    for _ in range(5):
        with torch.no_grad(), autocast(device_type='cuda'):
            _ = model(**inputs)

    # Measure time
    with torch.no_grad(), autocast(device_type='cuda'):
        for _ in range(10):
            start = time.time()
            _ = model(**inputs)
            end = time.time()
            times.append(end - start)

    avg_time = sum(times) / len(times)
    max_mem = None
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        print("CUDA is not available. Cannot retrieve GPU memory.")

    return avg_time * 1000, max_mem
