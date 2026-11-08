"""Start the service, send concurrent requests, report latency and memory.

Run with: uv run python scripts/load_test.py [--requests 300] [--image path.jpg]
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
from sklearn.datasets import get_data_home

HOST = "127.0.0.1"
PORT = 8799
BASE_URL = f"http://{HOST}:{PORT}"
MODELS = ["cnn", "pca-knn"]
CONCURRENCY = [1, 8]
STARTUP_TIMEOUT_SECONDS = 120


def rss_mb(pid: int) -> float:
    output = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=True
    )
    return int(output.stdout) / 1024


def default_image() -> Path:
    root = Path(get_data_home()) / "lfw_home" / "lfw_funneled"
    return next(iter(sorted(root.glob("*/*.jpg"))))


def wait_until_healthy(process: subprocess.Popen) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SystemExit("server exited during startup")
        try:
            httpx.get(f"{BASE_URL}/health", timeout=1).raise_for_status()
            return
        except httpx.HTTPError:
            time.sleep(0.2)
    raise SystemExit("server did not become healthy")


def run_load(model: str, concurrency: int, requests: int, image: bytes) -> dict:
    with httpx.Client(timeout=60) as client:

        def send(_: int) -> tuple[float, int]:
            started = time.perf_counter()
            response = client.post(
                f"{BASE_URL}/predict",
                params={"model": model},
                files={"file": ("face.jpg", image)},
            )
            return (time.perf_counter() - started) * 1000, response.status_code

        started = time.perf_counter()
        with ThreadPoolExecutor(concurrency) as pool:
            results = list(pool.map(send, range(requests)))
        elapsed = time.perf_counter() - started

    latencies = [latency for latency, _ in results]
    return {
        "p50": np.percentile(latencies, 50),
        "p95": np.percentile(latencies, 95),
        "rps": requests / elapsed,
        "errors": sum(status != 200 for _, status in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--image", type=Path, default=None)
    args = parser.parse_args()
    image = (args.image or default_image()).read_bytes()

    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", HOST,
         "--port", str(PORT), "--no-access-log"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=os.environ,
    )
    peak = [0.0]
    done = threading.Event()

    def sample_memory() -> None:
        while not done.wait(0.05):
            peak[0] = max(peak[0], rss_mb(server.pid))

    try:
        wait_until_healthy(server)
        idle = rss_mb(server.pid)
        sampler = threading.Thread(target=sample_memory)
        sampler.start()

        print("| Model | Concurrency | p50 (ms) | p95 (ms) | Requests/s | Errors |")
        print("|---|---|---|---|---|---|")
        for model in MODELS:
            for concurrency in CONCURRENCY:
                r = run_load(model, concurrency, args.requests, image)
                print(
                    f"| {model} | {concurrency} | {r['p50']:.1f} | {r['p95']:.1f} "
                    f"| {r['rps']:.1f} | {r['errors']} |"
                )
        done.set()
        sampler.join()
        print(f"\nServer RSS: idle {idle:.0f} MB, peak {peak[0]:.0f} MB (both models loaded)")
    finally:
        done.set()
        server.terminate()
        server.wait()


if __name__ == "__main__":
    main()
