import io
import multiprocessing as mp
import os
import time

import numpy as np
import speedtest
import torch
from PIL import Image


def _heavy_task(n):
    return sum(i * i for i in range(n))


def cpu_parallel_task():
    N = 10**6
    start = time.time()
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(_heavy_task, [N] * mp.cpu_count())
    duration = time.time() - start
    print(f"✅ CPU (parallel): {duration:.2f}s")


def ram_intensive_task():
    start = time.time()
    try:
        huge_array = np.ones((15000, 15000))  # ~1.8GB in float64
        _ = huge_array * 2
        duration = time.time() - start
        print(f"✅ RAM Intensive: {duration:.2f}s")
    except MemoryError:
        print("❌ RAM Intensive: MemoryError")


def gpu_test():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        start = time.time()
        a = torch.randn(10000, 10000, device=device)
        b = torch.randn(10000, 10000, device=device)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        duration = time.time() - start
        print(f"✅ GPU Matrix Mul: {duration:.2f}s")
    else:
        print("❌ GPU Test: CUDA device not available")


def disk_io_test():
    filename = "temp_io_test.dat"
    data = os.urandom(100 * 1024 * 1024)  # 100MB
    start = time.time()
    with open(filename, "wb") as f:
        f.write(data)
    write_time = time.time() - start

    start = time.time()
    with open(filename, "rb") as f:
        _ = f.read()
    read_time = time.time() - start

    os.remove(filename)
    print(f"✅ Disk I/O: Write {write_time:.2f}s | Read {read_time:.2f}s")


def network_test():
    try:
        st = speedtest.Speedtest()
        download = st.download() / 1e6
        upload = st.upload() / 1e6
        print(f"✅ Network: Download {download:.2f} Mbps | Upload {upload:.2f} Mbps")
    except Exception as e:
        print(f"❌ Network Test: {e}")


def image_render_test():
    try:
        img = Image.new("RGB", (10000, 10000), color="red")
        buf = io.BytesIO()
        start = time.time()
        img.save(buf, format="PNG")
        duration = time.time() - start
        print(f"✅ Image Render (CPU): {duration:.2f}s")
    except Exception as e:
        print(f"❌ Image Render: {e}")


def run_all():
    print("🔧 Running Benchmark Suite...")
    cpu_parallel_task()
    ram_intensive_task()
    gpu_test()
    disk_io_test()
    network_test()
    image_render_test()
    print("🏁 Benchmarking Complete.")


if __name__ == "__main__":
    run_all()
