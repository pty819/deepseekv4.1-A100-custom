"""Read-only safetensors mmap reader (numpy), independent of the dsv41 runtime package.

Only what the compression study needs: raw bytes of a tensor (or a row slice of it) without
copying the whole file. Never writes to the checkpoint.
"""
from __future__ import annotations

import json
import os
import struct

import numpy as np

DTYPE_BYTES = {"F8_E4M3": 1, "F8_E8M0": 1, "I8": 1, "U8": 1, "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8}


class Checkpoint:
    def __init__(self, path: str):
        self.path = path
        self.weight_map = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self._headers: dict[str, tuple[dict, int]] = {}
        self._mmaps: dict[str, np.memmap] = {}

    def _header(self, fn: str):
        if fn not in self._headers:
            with open(os.path.join(self.path, fn), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._headers[fn] = (json.loads(f.read(n)), 8 + n)
        return self._headers[fn]

    def _mmap(self, fn: str) -> np.memmap:
        if fn not in self._mmaps:
            self._mmaps[fn] = np.memmap(os.path.join(self.path, fn), dtype=np.uint8, mode="r")
        return self._mmaps[fn]

    def names(self, prefix: str = "") -> list[str]:
        return [n for n in self.weight_map if n.startswith(prefix)]

    def meta(self, name: str) -> tuple[str, list[int]]:
        hdr, _ = self._header(self.weight_map[name])
        m = hdr[name]
        return m["dtype"], m["shape"]

    def bytes(self, name: str, rows: slice | None = None) -> np.ndarray:
        """uint8 view [.., row_bytes] of the tensor (a copy, so the mmap page cache can be reclaimed)."""
        fn = self.weight_map[name]
        hdr, base = self._header(fn)
        m = hdr[name]
        dtype, shape = m["dtype"], list(m["shape"])
        start, end = m["data_offsets"]
        row_bytes = int(np.prod(shape[1:])) * DTYPE_BYTES[dtype] if len(shape) > 1 else DTYPE_BYTES[dtype]
        if rows is not None:
            r0, r1, _ = rows.indices(shape[0])
            start, end = start + r0 * row_bytes, start + r1 * row_bytes
            shape[0] = r1 - r0
        buf = np.array(self._mmap(fn)[base + start : base + end])
        return buf.reshape(shape[0], row_bytes) if len(shape) > 1 else buf

    def rows(self, name: str, idx: np.ndarray) -> np.ndarray:
        """uint8 [len(idx), row_bytes] for arbitrary row indices (one mmap slice per row)."""
        fn = self.weight_map[name]
        hdr, base = self._header(fn)
        m = hdr[name]
        dtype, shape = m["dtype"], list(m["shape"])
        start = m["data_offsets"][0]
        row_bytes = int(np.prod(shape[1:])) * DTYPE_BYTES[dtype]
        mm = self._mmap(fn)
        out = np.empty((len(idx), row_bytes), dtype=np.uint8)
        for j, r in enumerate(idx):
            o = base + start + int(r) * row_bytes
            out[j] = mm[o : o + row_bytes]
        return out
