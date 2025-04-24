import io
import multiprocessing as mp
import os
import time
import csv
import random
import psutil
import numpy as np
import torch
from PIL import Image
import subprocess

REPEAT = 5
CSV_LOG = True
CSV_FILE = "benchmark_results.csv"


def log_to_csv(name, duration):
    if CSV_LOG:
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, f"{duration:.4f}"])


def benchmark_avg(name, iterations=None):
    def decorator(func):
        def wrapper():
            durations = []
            it = iterations if iterations is not None else REPEAT
            for _ in range(it):
                result = func()
                if isinstance(result, float) and result > 0:
                    durations.append(result)
            if durations:
                avg_time = sum(durations) / len(durations)
                print(f"🟢 {name}: {avg_time:.2f}s (avg over {it})")
                log_to_csv(name, avg_time)
            else:
                print(f"🔴 {name}: Test skipped or failed.")

        return wrapper

    return decorator


def _heavy_task(n):
    return sum(i * i for i in range(n))


@benchmark_avg("CPU (single-thread)")
def cpu_single_task():
    N = 10**7
    start = time.time()
    _ = _heavy_task(N)
    return time.time() - start


@benchmark_avg("CPU (parallel)")
def cpu_parallel_task():
    N = 10**6
    start = time.time()
    with mp.Pool(mp.cpu_count()) as pool:
        _ = pool.map(_heavy_task, [N] * mp.cpu_count())
    return time.time() - start


@benchmark_avg("RAM (sequential)")
def ram_sequential_access():
    arr = np.ones((15000, 15000))  # ~1.8GB
    start = time.time()
    _ = arr * 2
    return time.time() - start


@benchmark_avg("RAM (random access)")
def ram_random_access():
    size = 10000000
    arr = np.random.rand(size)
    indices = [random.randint(0, size - 1) for _ in range(1000000)]
    start = time.time()
    _ = [arr[i] for i in indices]
    return time.time() - start


@benchmark_avg("GPU Matrix Multiply")
def gpu_test():
    if not torch.cuda.is_available():
        return None
    device = torch.device("cuda")
    a = torch.randn(10000, 10000, device=device)
    b = torch.randn(10000, 10000, device=device)
    _ = torch.matmul(a, b)  # Warm-up
    torch.cuda.synchronize()
    start = time.time()
    _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    return time.time() - start


@benchmark_avg("Disk I/O (sequential)")
def disk_io_test_seq():
    filename = "temp_seq.dat"
    data = os.urandom(500 * 1024 * 1024)  # 500MB
    start = time.time()
    with open(filename, "wb") as f:
        f.write(data)
    with open(filename, "rb") as f:
        _ = f.read()
    os.remove(filename)
    return time.time() - start


@benchmark_avg("Disk I/O (random)")
def disk_io_test_random():
    filename = "temp_rand.dat"
    block_size = 4096
    num_blocks = 1024 * 50
    data = os.urandom(block_size)
    with open(filename, "wb") as f:
        for _ in range(num_blocks):
            f.write(data)
    with open(filename, "rb") as f:
        offsets = [random.randint(0, num_blocks - 1) * block_size for _ in range(1000)]
        start = time.time()
        for offset in offsets:
            f.seek(offset)
            _ = f.read(block_size)
    os.remove(filename)
    return time.time() - start


@benchmark_avg("Network Speedtest", 1)
def network_test():
    try:
        result = subprocess.run(
            ["speedtest-cli", "--simple"], capture_output=True, text=True, timeout=30
        )
        print(result.stdout.strip())
        return 0.0  # Skip timing for now
    except Exception:
        return None


@benchmark_avg("Image Render")
def image_render_test():
    img = Image.new("RGB", (10000, 10000), color="red")
    buf = io.BytesIO()
    start = time.time()
    img.save(buf, format="PNG")
    return time.time() - start


def system_info():
    print("PcBenchmarkingSuite - System Info")
    print(f"CPU cores (logical): {psutil.cpu_count(logical=True)}")
    print(f"RAM: {round(psutil.virtual_memory().total / 1e9, 2)} GB")
    print(f"GPU: {'Available' if torch.cuda.is_available() else 'Unavailable'}")
    print()


def run_all():
    if CSV_LOG and os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)

    system_info()
    cpu_single_task()
    cpu_parallel_task()
    ram_sequential_access()
    ram_random_access()
    gpu_test()
    disk_io_test_seq()
    disk_io_test_random()
    network_test()
    image_render_test()
    print("✅ All benchmarks complete.")


if __name__ == "__main__":
    run_all()
