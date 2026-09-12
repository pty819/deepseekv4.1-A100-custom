"""Calibration-aware VQ encoder: pick the codebook entry that minimises the ACTIVATION-WEIGHTED error.

Plain VQ minimises sum_j (w_j - c_j)^2.  With the per-input-channel activation power h_j collected by
calib_collect.py, the right objective for a linear layer is sum_j h_j (w_j - c_j)^2 -- the diagonal of
the GPTQ/AWQ Hessian.  The codebook stays exactly the same (so the decoder and the stored format are
unchanged); only the encoder's choice changes, which is where the free accuracy is.

Full search over K codebook entries per group would be K/64 times more work than needed: for every one
of the 65536 possible 4-code tuples the M nearest entries under the unweighted metric are precomputed
once, and the weighted search runs over those M candidates.
"""
from __future__ import annotations

import numpy as np
import torch

from lossy import FP4


def build_candidates(codebook_codes: np.ndarray, topm: int = 64, device="cuda") -> torch.Tensor:
    """[65536, topm] int32: the topm nearest codebook entries of every 4-code tuple (unweighted)."""
    C = torch.from_numpy(FP4[codebook_codes]).float().to(device)            # [K, 4]
    u = torch.arange(65536, device=device)
    codes = torch.stack([(u >> (4 * j)) & 15 for j in range(4)], 1)          # packed byte-pair order
    V = torch.from_numpy(FP4).float().to(device)[codes]                      # [65536, 4]
    out = torch.empty(65536, topm, dtype=torch.int32, device=device)
    cn = (C * C).sum(1)
    for i in range(0, 65536, 4096):
        d = cn[None, :] - 2.0 * (V[i:i + 4096] @ C.T)
        out[i:i + 4096] = d.topk(topm, dim=1, largest=False).indices.to(torch.int32)
    return out


class CalibEncoder:
    """Weighted encoder for one codebook; call `encode(codes, h)` per (expert, matrix)."""

    def __init__(self, codebook_codes: np.ndarray, device, topm: int = 16, row_chunk: int = 512):
        self.dev = device
        self.codes = torch.from_numpy(codebook_codes).to(device)             # [K, 4] uint8
        self.C = torch.from_numpy(FP4[codebook_codes]).float().to(device)    # [K, 4]
        self.cand = build_candidates(codebook_codes, topm, device)
        self.topm, self.row_chunk = topm, row_chunk
        self.V = torch.from_numpy(FP4).float().to(device)

    @torch.inference_mode()
    def encode(self, codes: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """codes uint8 [N, K] (row-major nibbles), h float [K] -> re-quantized codes uint8 [N, K]."""
        N, K = codes.shape
        g = K // 4
        c4 = codes.view(N, g, 4).long()
        # the candidate table is indexed in packed byte-pair order (c0 lowest nibble)
        pk = c4[..., 0] | (c4[..., 1] << 4) | (c4[..., 2] << 8) | (c4[..., 3] << 12)
        hg = h.view(g, 4).to(self.dev)
        v = self.V[c4]                                                       # [N, g, 4]
        out = torch.empty_like(codes)
        for i in range(0, N, self.row_chunk):
            sl = slice(i, min(i + self.row_chunk, N))
            cd = self.cand[pk[sl]].long()                                    # [n, g, M]
            # sum_j h_j (C_j - v_j)^2, accumulated one coordinate at a time: the [n, g, M, 4]
            # intermediates never exist, which is what makes this affordable over a whole model
            d = None
            for j in range(4):
                cj = self.C[:, j][cd]                                        # [n, g, M]
                hj = hg[:, j][None, :, None]                                 # [1, g, 1]
                t = cj * (cj * hj - 2.0 * (v[sl][:, :, j] * hg[:, j])[:, :, None])
                d = t if d is None else d + t
            best = cd.gather(2, d.argmin(2, keepdim=True)).squeeze(2)        # [n, g]
            out[sl] = self.codes[best].view(-1, K)
        return out
