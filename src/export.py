"""Export CoreML + TFLite artifacts, generate embeddings binary, write manifest.

Usage:
  python -m src.export --ckpt artifacts/card_encoder.pt
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from src.config import (ARTIFACTS_DIR, EMBED_DIM, IMAGENET_MEAN, IMAGENET_STD,
                        IMAGES_DIR, INPUT_SIZE, MANIFEST_VERSION,
                        UNIQUE_ARTWORK_JSON)
from src.augment import build_eval_transform
from src.embed_binary import write_embeddings
from src.model import CardEncoder


class ExportWrapper(nn.Module):
    """Model + baked-in ImageNet normalization.

    Input:  NCHW float32 in [0,1]
    Output: NxD L2-normalized float32
    """
    def __init__(self, model: CardEncoder):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)


def sha256_file(path: Path) -> str:
    """Hash a file or directory (directory hashing retained for robustness)."""
    h = hashlib.sha256()
    if path.is_dir():
        for fpath in sorted(path.rglob("*")):
            if fpath.is_file():
                h.update(fpath.name.encode())
                with open(fpath, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
    else:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return h.hexdigest()


def export_coreml(wrapper: nn.Module, out_path: Path):
    import coremltools as ct
    wrapper.eval()
    example = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
    traced = torch.jit.trace(wrapper, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=example.shape)],
        outputs=[ct.TensorType(name="embedding")],
        convert_to="neuralnetwork",             # forces legacy .mlmodel format (iOS 11+)
        minimum_deployment_target=ct.target.iOS13,
    )
    mlmodel.save(str(out_path))


def export_tflite(wrapper: nn.Module, out_path: Path, tmp_dir: Path):
    """PyTorch -> ONNX -> TF SavedModel (onnx2tf) -> TFLite.

    onnx2tf unconditionally tries to download a calibration .npy file for any
    4D NHWC input with 3 channels.  Rather than monkey-patching its internals,
    we pre-seed the expected file path with dummy data so the download is
    skipped entirely (onnx2tf checks os.getcwd()/<filename> first).
    """
    import onnx
    import onnx2tf
    import tensorflow as tf

    tmp_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = tmp_dir / "encoder.onnx"
    tf_dir = tmp_dir / "tf_saved"
    wrapper.eval()
    example = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        wrapper, example, onnx_path,
        input_names=["image"], output_names=["embedding"],
        opset_version=17, do_constant_folding=True,
        dynamic_axes=None,
        dynamo=False,  # use legacy TorchScript exporter; dynamo needs onnxscript
    )
    onnx.checker.check_model(str(onnx_path))

    # Pre-seed calibration data so onnx2tf skips the network download.
    # onnx2tf looks for this file in os.getcwd() before attempting a download.
    calib_file_name = "calibration_image_sample_data_20x128x128x3_float32.npy"
    calib_file_path = Path.cwd() / calib_file_name
    if not calib_file_path.exists() or calib_file_path.stat().st_size < 1000:
        dummy_calib = np.zeros((20, 128, 128, 3), dtype=np.float32)
        np.save(str(calib_file_path), dummy_calib)

    try:
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(tf_dir),
            non_verbose=True,
        )
    except TypeError:
        # older/newer onnx2tf may not accept non_verbose
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(tf_dir),
        )

    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_dir))
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tfl = converter.convert()
    out_path.write_bytes(tfl)


@torch.no_grad()
def compute_all_embeddings(model: CardEncoder, scryfall_ids, device):
    transform = build_eval_transform()
    model.eval()
    embeds = np.empty((len(scryfall_ids), EMBED_DIM), dtype=np.float32)
    for i, sid in enumerate(tqdm(scryfall_ids, desc="embed all")):
        img = np.array(Image.open(IMAGES_DIR / f"{sid}.jpg").convert("RGB"))
        t = transform(image=img)["image"].unsqueeze(0).to(device)
        embeds[i] = model(t).cpu().numpy()[0]
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    embeds = embeds / np.clip(norms, 1e-8, None)
    return embeds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = CardEncoder().to(device)
    model.load_state_dict(ck["model"])
    wrapper = ExportWrapper(model).to(device).eval()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    coreml_path = ARTIFACTS_DIR / "card_encoder.mlmodel"
    tflite_path = ARTIFACTS_DIR / "card_encoder.tflite"
    embeds_path = ARTIFACTS_DIR / "card_embeds_v2.bin"
    manifest_path = ARTIFACTS_DIR / "manifest.json"

    print("exporting CoreML...")
    export_coreml(wrapper.cpu(), coreml_path)
    print("exporting TFLite...")
    export_tflite(wrapper.cpu(), tflite_path, ARTIFACTS_DIR / "_tmp_tflite")

    size_violations = []
    for pth in (coreml_path, tflite_path):
        if pth.is_dir():
            sz_bytes = sum(f.stat().st_size for f in pth.rglob("*") if f.is_file())
        else:
            sz_bytes = pth.stat().st_size
        sz_mb = sz_bytes / (1024 * 1024)
        print(f"{pth.name}: {sz_mb:.2f} MB")
        if sz_mb > 8.0:
            size_violations.append(f"{pth.name} exceeds 8 MB (got {sz_mb:.2f} MB)")
    if size_violations:
        for v in size_violations:
            print(f"SIZE VIOLATION: {v}", flush=True)
        raise SystemExit(f"Artifact size limit exceeded: {'; '.join(size_violations)}")

    encoder_sha = sha256_file(coreml_path)
    model_hash = int(encoder_sha[:8], 16)

    cards = json.loads(UNIQUE_ARTWORK_JSON.read_text())
    ids = [c["id"] for c in cards if (IMAGES_DIR / f"{c['id']}.jpg").exists()]
    print(f"embedding {len(ids)} cards...")
    model = model.to(device)
    embeds = compute_all_embeddings(model, ids, device)
    write_embeddings(embeds_path, ids, embeds, model_hash=model_hash,
                     version=MANIFEST_VERSION)

    manifest = {
        "version": MANIFEST_VERSION,
        "encoder": {
            "coreml": {"path": coreml_path.name, "sha256": encoder_sha},
            "tflite": {"path": tflite_path.name, "sha256": sha256_file(tflite_path)},
        },
        "embeddings": {
            "path": embeds_path.name,
            "sha256": sha256_file(embeds_path),
            "count": len(ids),
            "dim": EMBED_DIM,
        },
        "model_hash": f"0x{model_hash:08x}",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
