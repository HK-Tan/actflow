"""The set-form graft of Tan et al. (2026, arXiv:2608.29461) and the head-shared
batched graft tails. Graft at block l's output, every position:  x <- x - <x,u> u + t u."""
from __future__ import annotations

from contextlib import contextmanager

import torch

from .readout import run_tail, _out_tensor, _with_tensor


def graft_rows(x, u, t):
    """x [R,T,d], u [R,d] unit, t [R] -> grafted x (every position)."""
    coef = torch.einsum("rtd,rd->rt", x, u)
    return x - coef[..., None] * u[:, None, :] + t[:, None, None] * u[:, None, :]


@contextmanager
def graft_hook(model, layer, u, t):
    """The graft as a forward hook, as in Tan et al. (2026), one direction, for cross-checks."""
    def h(m, i, o):
        x = _out_tensor(o)
        vv = u.to(x.dtype)
        coef = (x @ vv).unsqueeze(-1)
        return _with_tensor(o, x - coef * vv + float(t) * vv)
    hd = model.model.layers[layer].register_forward_hook(h)
    try:
        yield
    finally:
        hd.remove()


@torch.no_grad()
def grafted_logits_rows(model, x, layer, mask, pos, lids, rows_item, U, T, batch=16, impl="slice", ids=None):
    """Letter logits [R,4] for R graft rows: row r grafts item rows_item[r] (an index into the chunk x
    [B,T,d] of block-l outputs) with direction U[r] [d] and target T[r]. The head (blocks 0..l) was computed
    once in x; only blocks l+1.. run here, `batch` rows at a time. impl='hook' needs the chunk's token ids [B,T]."""
    R = len(rows_item)
    out = torch.empty(R, len(lids), device=x.device)
    ri = torch.as_tensor(rows_item, device=x.device)
    for s in range(0, R, batch):
        rb = ri[s:s + batch]
        xr = graft_rows(x[rb], U[s:s + batch], T[s:s + batch])
        out[s:s + batch] = run_tail(model, xr, layer, mask[rb], pos[rb], lids, impl=impl, ids=None if ids is None else ids[rb])
    return out


@torch.no_grad()
def grafted_logits(model, x, layer, mask, pos, lids, U, T, batch=16, impl="slice"):
    """Grid version: every item in x [B,T,d] with every draft (U [D,d], T [D]) -> [B, D, 4]."""
    B, D = x.shape[0], U.shape[0]
    rows_item = torch.arange(B, device=x.device).repeat_interleave(D)
    di = torch.arange(D, device=x.device).repeat(B)
    return grafted_logits_rows(model, x, layer, mask, pos, lids, rows_item, U[di], T[di], batch, impl).view(B, D, -1)
