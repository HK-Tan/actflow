"""Draw bookkeeping and the pool-form reference (the graft of paper Eq. graft)."""
from __future__ import annotations

import numpy as np
import torch

from .data import permutation, draws


def all_draws(splits: dict, n_pool: int) -> dict[tuple[int, int], np.ndarray]:
    """{(k, d): pool indices}. Draws at one k partition the pool; every pool item is in one draw per k."""
    perm = permutation(n_pool, splits["perm_seed"])
    out = {}
    for k in splits["budgets"]:
        for d, idx in enumerate(draws(k, perm)):
            out[(k, d)] = idx
    return out


def curve_draws(splits: dict) -> set[tuple[int, int]]:
    return {(int(k), int(d)) for k, d in splits["curve_draws"]}


def manufactured(land, h_pool: np.ndarray) -> np.ndarray:
    """h*_{l,i} [n_pool, nL, d] of a landing file. 30_manufacture.py stores only the shift delta [draws, nL, d] of each draw,
    since every item of a draw gets the same shift, so h*_i = h_S,i + delta[draw of i], with h_pool the locked pool captures
    [n_pool, nL, d]. The honest stand-ins of 36_honest_reference.py store h_star itself, which is returned as is."""
    if "h_star" in getattr(land, "files", land):                        # an NpzFile or a dict of its arrays
        return land["h_star"]
    h = np.array(h_pool, np.float32)
    for dd, items in enumerate(land["draw_items"]):
        h[np.asarray(items)] += land["delta"][dd][None]
    return h


def reference(h_star: torch.Tensor, h_pool_mean: torch.Tensor, idx) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool-form reference at every layer. h_star [80, nL, d] landings, h_pool_mean [nL, d] = mean of the
    80 sandbag captures, idx = the draw. u_l = unit(mean_K h*_l - mean_U h_S,l) (+1e-6 in the norm, as
    build_directions), t_l = <mean_K h*_l, u_l>. Returns u [nL, d], t [nL]."""
    hk = h_star[torch.as_tensor(np.asarray(idx), device=h_star.device)].mean(0)   # [nL, d]
    dv = hk - h_pool_mean
    u = dv / (dv.norm(dim=-1, keepdim=True) + 1e-6)
    t = (hk * u).sum(-1)
    return u, t
