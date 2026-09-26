"""Scores and landing numbers."""
from __future__ import annotations

import numpy as np


def accuracy(ans, gold) -> float:
    return float(np.mean(np.asarray(ans) == np.asarray(gold)))


def balanced_accuracy(ans, gold, n=4) -> float:
    ans, gold = np.asarray(ans), np.asarray(gold)
    per = [np.mean(ans[gold == a] == a) for a in range(n) if (gold == a).any()]
    return float(np.mean(per))


def letter_hist(ans, n=4) -> list[float]:
    ans = np.asarray(ans)
    return [float(np.mean(ans == a)) for a in range(n)]


def gold_margin(logits, gold) -> np.ndarray:
    """gold logit minus the best other, per row."""
    logits, gold = np.asarray(logits, dtype=np.float64), np.asarray(gold)
    others = logits.copy(); others[np.arange(len(gold)), gold] = -np.inf
    return logits[np.arange(len(gold)), gold] - others.max(-1)


def binomial_se(p: float, n: int) -> float:
    return float(np.sqrt(max(p * (1 - p), 0.0) / n))


def last_layer_pair_bound(gS_i, gS_j, gH_i, gH_j, rho_i, rho_j) -> float:
    """Lower bound, over every shift shared by items i and j at the LAST layer, on max(|F_i - gH_i|_inf, |F_j - gH_j|_inf).
    At the last layer F_i(h_i + delta) = W'(h_i + delta) / rho(h_i + delta) with rho the RMS of the final norm, so
    F_i = w_i (a_i + u) with a_i = rho(h_i) gamma_S,i, w_i > 0 and u = W' delta shared (a superset of the reachable set).
    Eliminating u leaves e >= |nu Dl + theta (gH_i + gH_j) - gH_j|_inf with theta in [0, 1], nu >= 0 and
    Dl = rho_j gS_j - rho_i gS_i. The bound is the minimum of that over (theta, nu), an exact linear program."""
    from scipy.optimize import linprog                                       # laptop-only dependency (81_numbers, 80_paper)
    Dl = rho_j * gS_j - rho_i * gS_i
    A, b = [], []
    for a in range(len(Dl)):
        A.append([gH_i[a] + gH_j[a], Dl[a], -1.0]); b.append(gH_j[a])
        A.append([-(gH_i[a] + gH_j[a]), -Dl[a], -1.0]); b.append(-gH_j[a])
    res = linprog([0, 0, 1], A_ub=A, b_ub=b, bounds=[(0, 1), (0, None), (0, None)], method="highs")
    assert res.status == 0, res.message
    return float(res.fun)
