import struct
from pathlib import Path
import numpy as np
from src.config import BIN_MAGIC

HEADER_FMT = "<IIIII"   # magic, version, count, dim, model_hash
HEADER_SIZE = struct.calcsize(HEADER_FMT)
ID_BYTES = 36


def write_embeddings(path: Path, scryfall_ids, embeds: np.ndarray,
                     model_hash: int, version: int) -> None:
    """
    Write embeddings to a binary file with header and per-record structure.

    Validates:
    - embeds must be 2-D float32
    - id count matches embedding count
    - each id is exactly 36 characters (UUID format)
    - embeddings are L2-normalized (norm ~= 1.0, atol 1e-4)

    Args:
        path: Output file path
        scryfall_ids: List of UUID strings (36 chars each)
        embeds: (n, 256) float32 array, L2-normalized
        model_hash: 32-bit model hash
        version: Version number (typically 2)
    """
    if embeds.ndim != 2 or embeds.dtype != np.float32:
        raise ValueError("embeds must be 2-D float32")
    n, d = embeds.shape
    if n != len(scryfall_ids):
        raise ValueError("id count != embedding count")
    for sid in scryfall_ids:
        if len(sid) != ID_BYTES:
            raise ValueError(f"expected 36 char UUID, got {len(sid)} for {sid!r}")
    norms = np.linalg.norm(embeds, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("embeddings must be L2-normalized")

    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, BIN_MAGIC, version, n, d, model_hash & 0xFFFFFFFF))
        for sid, emb in zip(scryfall_ids, embeds):
            f.write(sid.encode("ascii"))
            f.write(emb.tobytes())


def read_embeddings(path: Path):
    """
    Read embeddings from a binary file.

    Returns:
        (ids, embeds, header) where:
        - ids: list of 36-char UUID strings
        - embeds: (n, 256) float32 array
        - header: dict with keys magic, version, count, dim, model_hash
    """
    with open(path, "rb") as f:
        header_raw = f.read(HEADER_SIZE)
        magic, version, count, dim, model_hash = struct.unpack(HEADER_FMT, header_raw)
        if magic != BIN_MAGIC:
            raise ValueError(f"bad magic 0x{magic:08x}")
        ids = []
        embeds = np.empty((count, dim), dtype=np.float32)
        for i in range(count):
            ids.append(f.read(ID_BYTES).decode("ascii"))
            embeds[i] = np.frombuffer(f.read(dim * 4), dtype=np.float32)
    return ids, embeds, {"magic": magic, "version": version, "count": count,
                         "dim": dim, "model_hash": model_hash}
