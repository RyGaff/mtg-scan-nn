import numpy as np
from pathlib import Path
import pytest
from src.embed_binary import write_embeddings, read_embeddings


def test_roundtrip(tmp_path):
    ids = [f"{i:08x}-0000-0000-0000-000000000000" for i in range(3)]
    embeds = np.random.randn(3, 256).astype(np.float32)
    embeds /= np.linalg.norm(embeds, axis=1, keepdims=True)
    model_hash = 0xDEADBEEF
    path = tmp_path / "embeds.bin"
    write_embeddings(path, ids, embeds, model_hash=model_hash, version=2)
    read_ids, read_embeds, header = read_embeddings(path)
    assert read_ids == ids
    np.testing.assert_allclose(read_embeds, embeds, atol=1e-6)
    assert header["magic"] == 0x4547544D
    assert header["version"] == 2
    assert header["count"] == 3
    assert header["dim"] == 256
    assert header["model_hash"] == 0xDEADBEEF


def test_on_disk_magic_is_ascii_MTGE(tmp_path):
    """First 4 bytes of the file must literally spell ASCII 'MTGE'."""
    ids = ["00000000-0000-0000-0000-000000000000"]
    embeds = np.zeros((1, 256), dtype=np.float32)
    embeds[0, 0] = 1.0
    path = tmp_path / "x.bin"
    write_embeddings(path, ids, embeds, model_hash=0, version=2)
    with open(path, "rb") as f:
        assert f.read(4) == b"MTGE"


def test_rejects_non_l2_normalized(tmp_path):
    ids = ["00000000-0000-0000-0000-000000000000"]
    bad = np.array([[1.0, 2.0] + [0.0]*254], dtype=np.float32)  # norm != 1
    with pytest.raises(ValueError, match="L2-normalized"):
        write_embeddings(tmp_path / "x.bin", ids, bad, model_hash=0, version=2)


def test_rejects_wrong_id_length(tmp_path):
    ids = ["short"]
    embeds = np.zeros((1, 256), dtype=np.float32)
    embeds[0, 0] = 1.0
    with pytest.raises(ValueError, match="36 char"):
        write_embeddings(tmp_path / "x.bin", ids, embeds, model_hash=0, version=2)
