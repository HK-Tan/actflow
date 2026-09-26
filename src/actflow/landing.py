"""Targets, the line, and the rules (paper Table 1): one-shot step, ActFlow (corrected or Euler), GD, each at full
rank or truncated (top-m, or top-round(PR) re-read at every step).

Every rule moves the k items of a draw TOGETHER by one common shift delta (the joint form): the per-item
readouts F_i are stacked into F_K: R^d -> R^{4k}, J_K = [J_1; ...; J_k] in R^{4k x d}, and
    one-shot step : delta = J_K(0)^+ (gamma_H - gamma_S)_K                                        (= ActFlow at N = 1)
    ActFlow       : delta_n = delta_{n-1} + R(delta_{n-1}) (gamma_K(t_n) - F_K(delta_{n-1})),  t_n = n/N     (corrected)
    Euler         : delta_n = delta_{n-1} + R(delta_{n-1}) (gamma_K(t_n) - gamma_K(t_{n-1}))                  (correction=False)
    GD            : delta <- delta + eta * sum_i J_i^T (gamma_H,i - F_i(delta)),  eta fixed, no backtracking
    R             : J_K^+ (full rank), J_{K,m}^+ (top-m singular directions, rank=m), or top-round(PR) at the current
                    point (rank="pr"); the rtol floor applies in every case
A draw of size k = 1 is the per-item rule: there is no separate per-item code path.
One ODE stands behind the walks, x' = R(J_K(x)) gamma_K'(t), x(0) = 0 (paper Eq. ode); ActFlow, Euler and one-shot are its
discretizations (paper Section 3.4, Algorithm 1).

All functions take a OneTokenTail `tail`, a layer l, batch tensors [B, ...] and `groups`, a list of index
tensors into the batch (one draw each). Nothing is polished: the landing error is measured, not corrected.
The 4k x 4k Gram solve is fp64.
"""
from __future__ import annotations

import torch


def raise_to(gamma, letter, m: float):
    """Target: logit `letter` set to max(others) + m, the other three held at gamma. gamma [B,4], letter [B]."""
    g = gamma.clone()
    B = g.shape[0]
    masked = g.clone()
    masked[torch.arange(B), letter] = -float("inf")
    g[torch.arange(B), letter] = masked.max(-1).values + m
    return g


def line_points(gamma_S, gamma_H, N: int):
    """The N points gamma(t_n), n = 1..N, of the straight line from gamma_S to gamma_H, as [N, B, 4]."""
    ts = torch.arange(1, N + 1, device=gamma_S.device, dtype=gamma_S.dtype) / N
    return gamma_S[None] + ts[:, None, None] * (gamma_H - gamma_S)[None]


def gram_stats(J):
    """rank and sigma_min/sigma_max of the per-item J [B,4,d] (fp64 SVD of the 4x4 Gram)."""
    G = torch.einsum("bad,bcd->bac", J.double(), J.double())
    ev = torch.linalg.eigvalsh(G).clamp(min=0).sqrt()                 # singular values of J
    smax = ev.max(-1).values
    rank = (ev > smax[:, None] * 1e-6).sum(-1)
    return rank, ev.min(-1).values / smax.clamp(min=1e-300)


def pinv_step(J, r):
    """Per-item least-norm step delta_i = J_i^T (J_i J_i^T)^+ r_i, [B,d], for J [B,4,d], r [B,4]. Gram solve
    in fp64, torch's default cut. Used only for the 'pooled per-item one shot' diagnostic of the joint step."""
    Jd = J.double()
    G = torch.einsum("bad,bcd->bac", Jd, Jd)
    w = torch.einsum("bac,bc->ba", torch.linalg.pinv(G, hermitian=True), r.double())
    return torch.einsum("bad,ba->bd", Jd, w).to(J.dtype)


def participation_ratio(ev):
    """PR = (sum ev)^2 / sum ev^2 for ev = sigma^2 (the Gram eigenvalues): how many directions carry the energy."""
    return float(ev.sum() ** 2 / (ev ** 2).sum().clamp(min=1e-300))


def _kept_basis(Jg, rtol=1e-4, rank=None):
    """fp64 eigh of the Gram of one stacked J_g [4k, d]. Returns (U_m [4k, m], ev_m [m], ev_all [4k] ascending).
    keep = (ev > rtol^2 * ev_max)  AND  top-`rank` (int) or top-round(PR) ("pr"). The rtol floor is a numerical
    guard: where J_g loses full row rank (the shared-shift map is not a submersion there, e.g. at the last layer
    on Qwen 7B) some sigma are zero up to floating-point error, and dividing by them blows up the least-norm step."""
    ev, U = torch.linalg.eigh(Jg @ Jg.T)                    # ascending; ev = sigma^2 of J_g
    ev = ev.clamp(min=0)
    keep = ev > ev[-1] * rtol ** 2
    if rank is not None:
        m = int(round(participation_ratio(ev))) if rank == "pr" else int(rank)
        m = max(1, min(m, len(ev)))
        keep[: len(ev) - m] = False                         # top-m eigenpairs only
    return U[:, keep], ev[keep], ev


def stacked_pinv_step(J, r, groups, rtol=1e-4, rank=None):
    """Least-norm common step per group. J [B,4,d], r [B,4] -> delta [B,d] (rows of one group are equal).
    Per group: J_g = J[idx].reshape(4k, d), delta_g = J_g^T U_m (U_m^T r_g / ev_m) = J_{g,m}^+ r_g, Gram solve in fp64.
    rank None: full rank (rtol floor only). rank m: the least-norm minimizer of ||J_g d - r_g|| among steps confined
    to span(v_1..v_m); it does NOT land for m < 4k (the residual on the discarded output directions is deliberately
    never followed). rank "pr": m = round PR of the spectrum at this point."""
    out = torch.zeros_like(J[:, 0])
    for idx in groups:
        Jg = J[idx].reshape(-1, J.shape[-1]).double()          # [4k, d]
        rg = r[idx].reshape(-1).double()                        # [4k]
        Um, evm, _ = _kept_basis(Jg, rtol, rank)
        w = Um @ ((Um.T @ rg) / evm)
        out[idx] = (Jg.T @ w).to(J.dtype)[None]
    return out


def stacked_spectrum(J, groups, top=3, rtol=1e-4):
    """Full singular spectrum of J_stack per group (descending, [4k]) and the top right-singular input
    directions v_a = J_g^T u_a / sigma_a ([top, d]). fp64 eigh of the Gram. A row is a unit vector (up to fp error)
    only where sigma_a > rtol * sigma_1 (valid, the cut stacked_gram_stats counts); at or below that floor u_a spans the
    numerical null space (at the last layer J_K has rank <= 4 + k, sigma_a / sigma_1 ~ 1e-8) and the row is set to zero.
    Returns (list of sigma [4k] fp32, list of v [top, d] fp32, list of valid [top] bool)."""
    sigmas, vs, oks = [], [], []
    for idx in groups:
        Jg = J[idx].reshape(-1, J.shape[-1]).double()
        ev, U = torch.linalg.eigh(Jg @ Jg.T)
        sig = ev.clamp(min=0).sqrt().flip(0)
        Ud = U.flip(1)
        ok = sig[:top] > sig[0] * rtol
        v = torch.where(ok[None], (Jg.T @ Ud[:, :top]) / sig[:top].clamp(min=1e-30), 0.0)
        sigmas.append(sig.float()); vs.append(v.T.float()); oks.append(ok)
    return sigmas, vs, oks


def stacked_gram_stats(J, groups, rtol=1e-4):
    """Retained rank (singular values above rtol * sigma_max, the same cut stacked_pinv_step uses) and
    sigma_min/sigma_max of J_stack per group (fp64 SVD of the 4k x 4k Gram). Returns lists."""
    ranks, ratios = [], []
    for idx in groups:
        Jg = J[idx].reshape(-1, J.shape[-1]).double()
        ev = torch.linalg.eigvalsh(Jg @ Jg.T).clamp(min=0).sqrt()
        smax = ev.max()
        ranks.append(int((ev > smax * rtol).sum())); ratios.append(float(ev.min() / smax.clamp(min=1e-300)))
    return ranks, ratios


def oneshot(tail, h0, layer, gamma_H, groups, F0=None, J0=None, rtol=1e-4, rank=None):
    """h_i* = h_i + delta_g, delta_g = J_stack(0)^+ (gamma_H - F(h))_stack. Pass F0, J0 to reuse them.
    rank: truncated one-shot step (see stacked_pinv_step). Equals actflow() at N = 1."""
    if F0 is None:
        F0, J0 = tail.FJ(h0, layer)
    return h0 + stacked_pinv_step(J0, gamma_H - F0, groups, rtol, rank=rank)


REC_KEYS = ["e", "e_kept", "e_dropped", "step", "shift", "share0", "share3", "label_acc", "m", "pr"]   # _record_step's scalar keys (+ sigma [4k])
GD_KEYS = ["loss", "err", "eta", "T"]                                                                  # gd()'s per-step keys


def _record_step(logged, n, F, J, h, h_prev, h0, gamma_n, groups, rtol, rank, gold, V0):
    """Append one step's diagnostics per group (state x_n, spectrum read AT x_n). gamma_n = gamma(t_n), gamma_S at n = 0.
    logged[gi] holds lists over n = 0..N of
        sigma [4k]          singular values of J_K(x_n)
        m, pr               kept count (rule + rtol floor) and participation ratio at x_n
        e, e_kept, e_dropped   ||e_n||, ||U_m^T e_n||, ||e_n - U_m U_m^T e_n||, e_n = gamma(t_n) - F(x_n), U_m at x_n
        step, shift         ||x_n - x_{n-1}||, ||x_n||
        share0, share3      ||V0^T x_n||^2 / ||x_n||^2 with V0 the rule's kept right-singular directions at x_0,
                            and the same with the top-3 at x_0
        label_acc           accuracy on the k labelled items at x_n (correct letter on top); only when gold is given"""
    for gi, idx in enumerate(groups):
        Jg = J[idx].reshape(-1, J.shape[-1]).double()
        Um, evm, ev = _kept_basis(Jg, rtol, rank)
        e = (gamma_n[idx] - F[idx]).reshape(-1).double()
        ek = Um.T @ e
        x = (h[idx[0]] - h0[idx[0]]).double()
        L = logged.setdefault(gi, {})
        if n == 0:                                          # right singular directions at x_0, for the shares
            V0[gi] = (Jg.T @ Um) / evm.sqrt().clamp(min=1e-30)          # [d, m0], columns = v_a of the kept set
            ev_all, U_all = torch.linalg.eigh(Jg @ Jg.T)                # ascending
            V0[(gi, 3)] = (Jg.T @ U_all[:, -3:]) / ev_all[-3:].clamp(min=1e-30).sqrt()   # top-3 v_a
        xn = float(x.norm())
        rec = dict(
            sigma=ev.flip(0).sqrt().float().cpu(),
            m=int(Um.shape[1]),
            pr=participation_ratio(ev),
            e=float(e.norm()), e_kept=float(ek.norm()), e_dropped=float((e - Um @ ek).norm()),
            step=float((h[idx[0]] - h_prev[idx[0]]).norm()), shift=xn,
            share0=float((V0[gi].T @ x).norm() ** 2 / xn ** 2) if xn > 0 else 0.0,
            share3=float((V0[(gi, 3)].T @ x).norm() ** 2 / xn ** 2) if xn > 0 else 0.0,
        )
        if gold is not None:
            rec["label_acc"] = float((F[idx].argmax(-1) == gold[idx]).double().mean())
        for k_, v_ in rec.items():
            L.setdefault(k_, []).append(v_)


def actflow(tail, h0, layer, points, groups, rtol=1e-4, rank=None, correction=True, logged=None, gold=None):
    """Walk along points [N,B,4] (paper Eq. actflow); one batched FJ per step, plus one at the end when logging.
    correction=True : x_n = x_{n-1} + R (gamma(t_n) - F(x_{n-1}))       the corrected step (the code of the paper)
    correction=False: x_n = x_{n-1} + R (gamma(t_n) - gamma(t_{n-1}))   Euler
    The two coincide at n = 1 (gamma(t_0) = F(x_0)), so at N = 1 both are the one-shot step.
    rank: None | int | "pr" -- ActFlow_m confines every step to the CURRENT kept singular directions (re-computed
    at each point, so they can rotate along the path) and does not land for m < 4k.
    logged: dict to fill per step (the per-step record, saved as logged_* arrays) (see _record_step). gold [B]: correct letters, for label_acc."""
    h, h_prev = h0, h0
    prev = None                                             # gamma(t_{n-1}); at n = 0 it is F(h0) = gamma_S
    N = points.shape[0]
    V0 = {}
    for n in range(N + 1):
        if n == N and logged is None:
            break
        F, J = tail.FJ(h, layer)
        if prev is None:
            prev = F
        if logged is not None:
            _record_step(logged, n, F, J, h, h_prev, h0, prev, groups, rtol, rank, gold, V0)
        if n == N:
            break
        r = points[n] - (F if correction else prev)
        h_prev = h
        h = h + stacked_pinv_step(J, r, groups, rtol, rank=rank)
        prev = points[n]
    return h


def gd(tail, h0, layer, gamma_H, groups, tol=0.5, max_steps=1000, first_step=0.01, logged=None):
    """GD: fixed-step descent on L_g(delta) = sum_{i in g} 1/2 ||F(h_i + delta) - gamma_H,i||^2 in the common shift
    delta_g of each group. One batched vjp per step (g_i = J_i^T r_i, summed within the group).
    Knobs, declared and never tuned:
        eta        set once per group so that the FIRST step has length first_step * mean_i ||h_i||; fixed after that
        max_steps  the budget (1000); T = n * eta is the elapsed time of gradient flow
        tol        a group stops when every item has ||gamma_H,i - F_i||_inf < tol (0.5 = the 'landed' threshold)
    No backtracking: every step is taken, and the loss is logged so a rise is visible. Returns h [B,d], steps [B],
    F [B,4].
    logged[gi]: per step, loss, err = max_i ||r_i||_inf, eta, T."""
    B = h0.shape[0]
    h = h0.clone()
    steps = torch.zeros(B, dtype=torch.long, device=h0.device)
    done = torch.zeros(B, dtype=torch.bool, device=h0.device)
    eta = {}
    for it in range(max_steps + 1):
        F, g = tail.vjp(h, layer, lambda Fc: gamma_H - Fc)          # g_i = J_i^T (gamma_H,i - F_i)
        r = gamma_H - F
        err = r.abs().max(-1).values
        step = torch.zeros_like(h)
        for gi, idx in enumerate(groups):
            if bool(done[idx].all()):
                continue
            gs = g[idx].sum(0)                                      # [d]
            if gi not in eta:
                eta[gi] = float(first_step * h0[idx].norm(dim=-1).mean() / gs.norm().clamp(min=1e-12))
            if logged is not None:
                L = logged.setdefault(gi, {})
                for k_, v_ in dict(loss=float(0.5 * (r[idx] ** 2).sum()), err=float(err[idx].max()),
                                   eta=eta[gi], T=it * eta[gi]).items():
                    L.setdefault(k_, []).append(v_)
            if bool((err[idx] < tol).all()) or it == max_steps:
                done[idx] = True
                continue
            step[idx] = (eta[gi] * gs)[None]
            steps[idx] += 1
        if bool(done.all()):
            break
        h = h + step
    return h, steps, tail.F(h, layer)


def landing_numbers(F, gamma_H, gold, tol=0.5):
    """landing error (l_inf), realised gold margin, landed = err < tol, gold_first = gold wins the argmax;
    F, gamma_H [B,4], gold [B]. A non-finite row is never landed and never gold_first (argmax of NaN is letter A)."""
    B = F.shape[0]
    err = (F - gamma_H).abs().max(-1).values
    others = F.clone(); others[torch.arange(B), gold] = -float("inf")
    margin = F[torch.arange(B), gold] - others.max(-1).values
    return err, margin, err < tol, (F.argmax(-1) == gold) & torch.isfinite(F).all(-1)


def seq_points(gamma_S, gamma_H, N: int, groups, orders):
    """The sequential path, [N, B, 4]: inside each group the items move one at a time, in the order `orders[g]`
    (a permutation of the group's k local positions), each over N // k consecutive steps, linearly from
    gamma_S to gamma_H. Items not yet reached sit at gamma_S, finished items at gamma_H, so gamma(t_N) = gamma_H.
    Needs N % k == 0 for every group. At k = 1 it is line_points."""
    pts = gamma_S[None].repeat(N, 1, 1).clone()
    ns = torch.arange(1, N + 1, device=gamma_S.device, dtype=gamma_S.dtype)
    for idx, order in zip(groups, orders):
        k = len(idx)
        assert N % k == 0, f"sequential path needs N % k == 0 (N={N}, k={k})"
        s = N // k
        for j, pos in enumerate(order):
            i = idx[int(pos)]
            a = ((ns - j * s) / s).clamp(0, 1)                                # 0 before its turn, 1 after
            pts[:, i] = gamma_S[i][None] + a[:, None] * (gamma_H[i] - gamma_S[i])[None]
    return pts


def seq_orders(k: int, draw_ids, seed: int):
    """One item order per draw: rng(seed + 1000 k + d).permutation(k)."""
    import numpy as np
    return [np.random.default_rng(seed + 1000 * k + int(d)).permutation(k) for d in draw_ids]
