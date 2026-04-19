"""Measure CoreML + TFLite single-image latency.

Usage:
  python -m src.benchmark

Writes artifacts/benchmark_results.json.
"""
import argparse
import json
import platform
import time
import numpy as np
from pathlib import Path
from src.config import ARTIFACTS_DIR, INPUT_SIZE


def bench_coreml(path: Path, n: int = 100) -> float:
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
    return dt


def bench_tflite(path: Path, n: int = 100) -> float:
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
    return dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    args = p.parse_args()
    coreml_ms = bench_coreml(ARTIFACTS_DIR / "card_encoder.mlmodel", args.n)
    tflite_ms = bench_tflite(ARTIFACTS_DIR / "card_encoder.tflite", args.n)

    out = ARTIFACTS_DIR / "benchmark_results.json"
    out.write_text(json.dumps({
        "n_iterations": args.n,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "coreml_ms_per_image": coreml_ms,
        "tflite_ms_per_image": tflite_ms,
        "note": "host-CPU proxy; on-device latency differs",
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
