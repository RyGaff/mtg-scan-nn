"""Measure CoreML + TFLite single-image latency.

Usage:
  python -m src.benchmark
"""
import argparse
import time
import numpy as np
from pathlib import Path
from src.config import ARTIFACTS_DIR, INPUT_SIZE

def bench_coreml(path: Path, n: int = 100):
    import coremltools as ct
    model = ct.models.MLModel(str(path))
    x = np.random.rand(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
    for _ in range(5):
        model.predict({"image": x})
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict({"image": x})
    dt = (time.perf_counter() - t0) * 1000 / n
    print(f"CoreML (host CPU, sim proxy): {dt:.2f} ms/image")

def bench_tflite(path: Path, n: int = 100):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    x = np.random.rand(*inp["shape"]).astype(np.float32)
    for _ in range(5):
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        interp.get_tensor(out["index"])
    t0 = time.perf_counter()
    for _ in range(n):
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        interp.get_tensor(out["index"])
    dt = (time.perf_counter() - t0) * 1000 / n
    print(f"TFLite (host CPU): {dt:.2f} ms/image")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    args = p.parse_args()
    bench_coreml(ARTIFACTS_DIR / "card_encoder.mlmodel", args.n)
    bench_tflite(ARTIFACTS_DIR / "card_encoder.tflite", args.n)

if __name__ == "__main__":
    main()
