"""Every number quoted in the prose of the paper's Results, Discussion and Appendices A-B that 80_paper.py does not print
(laptop, no GPU, a few seconds).
Reads
  results*/graft/{organism}/{stem}_k{k}/k{k'}_d{d}.json    ActFlow and GD grafts (35_graft.py): lhat, sel_acc, test answers
  results*/honest-graft/{organism}/honestk_N0_k{k}/...      the honest graft (run_honest_ref.sh)
  results*/landing/{organism}/{stem}_k{k}.npz               landings (30_manufacture.py): delta, landed, logged_m, ...
  results*/landing/{organism}/seed_k{k}.npz                 the stacked Jacobian at x = 0: rank_stack, sigma_stack, gamma_S, gamma_H
  results*/captures/{organism}/pool.npz, honest_pool.npz    locked and honest residual streams of the pool U (15_honest.py)
  results*/sft/{organism}/sft_k{k}.json                     fine-tuning (70_sft.py)
Prints one line per number: where it is in main.tex, the paper's value, the value recomputed here, and a flag when they
differ. Layers are printed 1-based as in the paper (layer l = index l-1 of the result arrays)."""
import glob, json, sys
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from pathlib import Path
import numpy as np

A = Path(__file__).resolve().parents[1]                                # the actflow repo
sys.path.insert(0, str(A / "src"))
from actflow.metrics import last_layer_pair_bound
from actflow.reference import manufactured

ROOT = {"qwen": A / "results_qwen7b", "llama": A / "results_llama8b", "mistral": A / "results_mistral7b"}
EPS = {"qwen": 1e-6, "llama": 1e-5, "mistral": 1e-5}                    # RMSNorm constant of the final norm
SIX = [(m, o) for m in ROOT for o in ("prompt", "ft_lock")]              # the six locked models of Table 2
TEN = SIX + [("qwen", f"ft_lock_s{s}") for s in range(2, 6)]             # + the Qwen LoRA locks of seeds 2 to 5
KS = [4, 10, 40]
RULES = ("af5_N40", "afpr_N40", "af_N40")                                # AF_5, AF_PR, AF_full (GN, N = 40, linear path)
Q = 4                                                                    # answer letters


def f2(x, nd=2):
    """Half-up rounding, as the tables print."""
    return str(Decimal(str(round(float(x), 6))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def say(tag, paper, here, ok=None, note=None):
    """ok: whether `here` supports the paper's value (default: the two strings are equal); note: a caveat to flag."""
    ok = (str(paper) == str(here)) if ok is None else ok
    print(f"{tag:<62} paper {str(paper):<22} here {here}{'' if ok else '   <-- DIFFERS'}{f'   <-- note: {note}' if note else ''}")


@lru_cache(None)
def recs(m, org, stem, k, kp=None):
    gdir = ROOT[m] / ("honest-graft" if stem.startswith("honest") else "graft") / org / f"{stem}_k{k}"
    r = [json.load(open(f)) for f in glob.glob(str(gdir / f"k{kp or k}_d*.json"))]
    return sorted(r, key=lambda x: x["draw"])


def acc(m, org, stem, k, key="test_acc"):
    return float(np.mean([x[key] for x in recs(m, org, stem, k)]))


def sft(m, org, k, key="test", ck="1024"):
    return float(json.load(open(ROOT[m] / "sft" / org / f"sft_k{k}.json"))["mean_by_checkpoint"][ck][key])


def ref(m, org, key):                                                    # locked / honest accuracy, as in 80_paper
    return recs(m, org, "af5_N40", 40)[0][key]


def landing(m, org, stem):
    return np.load(ROOT[m] / "landing" / org / f"{stem}.npz")


@lru_cache(None)
def caps(m, org):
    c = ROOT[m] / "captures" / org
    return np.load(c / "pool.npz")["h"], np.load(c / "honest_pool.npz")["h"]


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)        # as actflow.reference


def w_af(m, org, stem, k):
    """Manufactured grafting directions [draws, nL, d]: unit(mean_K h* - mean_U h_S) (Eq. graft)."""
    hS = caps(m, org)[0]
    hs, mS = manufactured(landing(m, org, f"{stem}_k{k}"), hS), hS.mean(0)
    return np.stack([unit(hs[np.asarray(r["items"])].mean(0) - mS) for r in recs(m, org, stem, k)])


def w_hon(m, org, k):
    """Harvested (honest-graft) directions [draws, nL, d]: unit(mean_K h_hon - mean_U h_S), on the draws of AF_5."""
    hS, hH = caps(m, org)
    return np.stack([unit(hH[np.asarray(r["items"])].mean(0) - hS.mean(0)) for r in recs(m, org, "af5_N40", k)])


def disp(x):
    return float(f2(x))


# ---------------- Section 5.1 (sec:q1): means over the six locked models ------------------------------------------------
print("== sec:q1 (ARC-Easy test accuracy, mean over the six locked models unless noted)")
mean6 = lambda f: float(np.mean([f(m, o) for m, o in SIX]))
say("q1 k=40 AF5", "0.85", f2(mean6(lambda m, o: acc(m, o, "af5_N40", 40))))
say("q1 k=40 locked models", "0.05", f2(mean6(lambda m, o: ref(m, o, "acc_locked"))))
say("q1 k=40 honest models", "0.92", f2(mean6(lambda m, o: ref(m, o, "acc_honest"))))
say("q1 k=40 FT", "0.88", f2(mean6(lambda m, o: sft(m, o, 40))))
gap = {(m, o): abs(disp(sft(m, o, 40)) - disp(acc(m, o, "af5_N40", 40))) for m, o in SIX}
say("q1 AF5 within 0.05 of FT exc. Mistral prompt: max |FT-AF5| of 5", "<= 0.05",
    f2(max(v for c, v in gap.items() if c != ("mistral", "prompt"))), max(v for c, v in gap.items() if c != ("mistral", "prompt")) <= 0.05 + 1e-9)
say("q1 Mistral prompt AF5 vs FT (k=40)", "0.69 vs 0.88", f"{f2(acc('mistral', 'prompt', 'af5_N40', 40))} vs {f2(sft('mistral', 'prompt', 40))}")
hg = {(m, o): abs(disp(acc(m, o, "af5_N40", 40)) - disp(acc(m, o, "honestk_N0", 40))) for m, o in SIX}
say("q1 AF5 within 0.06 of honest graft (k=40): locks", "5", str(sum(v <= 0.06 + 1e-9 for v in hg.values())),
    note="Mistral prompt is 0.06 as printed, 0.065 raw")
print("   |AF5 - honest graft| per lock (as printed):", {f"{m}/{o}": f2(v) for (m, o), v in hg.items()},
      "| raw, Mistral prompt:", f2(acc("mistral", "prompt", "af5_N40", 40) - acc("mistral", "prompt", "honestk_N0", 40), 3))
say("q1 k=40 OpenBookQA AF5", "0.68", f2(mean6(lambda m, o: acc(m, o, "af5_N40", 40, "obqa_test_acc"))))
say("q1 k=40 OpenBookQA FT", "0.73", f2(mean6(lambda m, o: sft(m, o, 40, "test_obqa"))))
say("q1 k=40 OpenBookQA honest models", "0.79", f2(mean6(lambda m, o: ref(m, o, "obqa_acc_honest"))))
gd = {k: (mean6(lambda m, o: acc(m, o, "gd_N1000", k)), mean6(lambda m, o: acc(m, o, "af5_N40", k))) for k in (4, 40)}
say("q1 GD below AF5 at k=40 (GD vs AF5)", "below", f"{'below' if gd[40][0] < gd[40][1] else 'above'} ({f2(gd[40][0])} vs {f2(gd[40][1])})",
    gd[40][0] < gd[40][1])
say("q1 GD above AF5 at k=4 (GD vs AF5)", "above", f"{'above' if gd[4][0] > gd[4][1] else 'below'} ({f2(gd[4][0])} vs {f2(gd[4][1])})",
    gd[4][0] > gd[4][1])
d_pr = {(m, o): [disp(acc(m, o, "af5_N40", k)) - disp(acc(m, o, "afpr_N40", k)) for k in KS] for m, o in SIX}
oth = max(abs(v) for c, x in d_pr.items() if c != ("qwen", "ft_lock") for v in x)
say("q1 AFPR closely matches AF5 exc. Qwen LoRA: max |AF5-AFPR| other 5", "(closely)", f2(oth), True)
print("   AF5 - AFPR at k=4,10,40 (as printed):", {f"{m}/{o}": [f2(v) for v in x] for (m, o), x in d_pr.items()})
say("q1 Qwen LoRA AF5-AFPR at k=4,10,40 (>0 even at k=40)", "> 0", ", ".join(f2(v) for v in d_pr[("qwen", "ft_lock")]),
    d_pr[("qwen", "ft_lock")][-1] > 0)

# ---------------- Section 5.2 (sec:q2): exact landing -------------------------------------------------------------------
print("== sec:q2")
n = ok = 0
for m, o in SIX:
    for k in KS:
        al = landing(m, o, f"af_N40_k{k}")["all_landed"]
        for r in recs(m, o, "af_N40", k):
            n += 1; ok += bool(al[r["draw"], r["lhat"]])
            if not al[r["draw"], r["lhat"]]: miss = f"{m}/{o} k={k} draw {r['draw']} layer {r['lhat'] + 1}"
say("q2 AF_full all q logits within 0.5 at lhat, draws k>=4", "179/180", f"{ok}/{n}")
print("   the draw that does not land:", miss)
raw = sum(acc(m, o, "af5_N40", k) > acc(m, o, "af_N40", k) for m, o in SIX for k in KS)
shown = sum(disp(acc(m, o, "af5_N40", k)) > disp(acc(m, o, "af_N40", k)) for m, o in SIX for k in KS)
say("q2 AF5 > AF_full on ARC, rows of Table 2 (raw means)", "16/18", f"{raw}/18")
say("q2 AF5 > AF_full on ARC, rows of Table 2 (as printed)", "16/18", f"{shown}/18")
for stem, pap in (("af_N40", "0.92/0.61"), ("af5_N40", "0.84/0.85")):
    lab = mean6(lambda m, o: np.mean([r["sel_acc"][r["lhat"]] for r in recs(m, o, stem, 40)]))
    say(f"q2 k=40 labeled/test accuracy {stem}", pap, f"{f2(lab)}/{f2(mean6(lambda m, o: acc(m, o, stem, 40)))}")

# ---------------- Section 5.3 (sec:q3): one-item shift memorizes the letter ---------------------------------------------
print("== sec:q3")
gold = np.load(ROOT["qwen"] / "captures" / "ft_lock" / "pool.npz")["gold"]
r1 = recs("qwen", "ft_lock", "af_N40", 1)                                # one-item AF_5 = AF_full (4 Jacobian rows)
same = np.mean([np.sum(np.array(r["test_answers"]) == gold[r["items"][0]]) for r in r1])
say("q3 Qwen LoRA one-item AF5: test answers = labeled letter", "65 of 100", f"{same:.0f} of 100")
print(f"   mean over the {len(r1)} one-item draws: {same:.2f}")

# ---------------- Section 5.4 (sec:q4): manufactured against harvested directions ---------------------------------------
print("== sec:q4")
cs = []
for m, o in SIX:
    for k in KS:
        WH = w_hon(m, o, k)
        for stem in RULES:
            W = w_af(m, o, stem, k)
            for r in recs(m, o, stem, k):
                assert r["items"] == recs(m, o, "af5_N40", k)[r["draw"]]["items"]
                if r["lhat"] is not None:
                    cs.append(abs(float(W[r["draw"], r["lhat"]] @ WH[r["draw"], r["lhat"]])))
say("q4 draws of the three rules with k>=4", "540", str(len(cs)))
say("q4 median |cos(manufactured, harvested)| at lhat", "0.05", f2(np.median(cs)))
dims = {m: caps(m, "prompt")[0].shape[-1] for m in ROOT}
say("q4 random direction E|cos| = sqrt(2/(pi d))", "0.01", "/".join(sorted({f2(np.sqrt(2 / (np.pi * d))) for d in dims.values()})))
print("   d:", dims, "values", [round(float(np.sqrt(2 / (np.pi * d))), 4) for d in dims.values()])
caf, chon = [], []
for m, o in SIX:
    W, WH = w_af(m, o, "af5_N40", 40), w_hon(m, o, 40)
    for l in sorted({r["lhat"] for r in recs(m, o, "af5_N40", 40)}):
        caf.append(float(W[0, l] @ W[1, l])); chon.append(float(WH[0, l] @ WH[1, l]))
say("q4 k=40 draw-to-draw cos at either draw's lhat, AF5", "0.71 to 0.92", f"{f2(min(caf))} to {f2(max(caf))}")
say("q4 k=40 draw-to-draw cos at either draw's lhat, honest graft", ">= 0.95", f2(min(chon)), disp(min(chon)) >= 0.95)
fails = [("mistral", "ft_lock"), ("qwen", "ft_lock_s5")]
say("q4 AF5 k=40 on Mistral LoRA, Qwen LoRA seed 5", "0.72 and 0.82", " and ".join(f2(acc(m, o, "af5_N40", 40)) for m, o in fails))
cmax = max(abs(float(w_af(m, o, "af5_N40", 40)[r["draw"], r["lhat"]] @ w_hon(m, o, 40)[r["draw"], r["lhat"]]))
           for m, o in fails for r in recs(m, o, "af5_N40", 40))
say("q4 there: max |cos| AF5 vs harvested at lhat (k=40)", "<= 0.05", f2(cmax), disp(cmax) <= 0.05)
hmax = max(acc(m, o, "honestk_N0", 40) for m, o in fails)
say("q4 there: honest graft k=40 (max of the two locks)", "<= 0.03", f2(hmax), disp(hmax) <= 0.03,
    "0.02 at k=40, 0.03 only as the max over k=4,10,40" if f2(hmax) != "0.03" else None)
print("   honest graft at k=4,10,40:", {f"{m}/{o}": [f2(acc(m, o, "honestk_N0", k), 3) for k in KS] for m, o in fails})

# ---------------- Section 5.5 (sec:general) -----------------------------------------------------------------------------
print("== sec:general (Qwen prompt and LoRA locks, AF5 and AFPR, k = 4, 10, 40)")
QW = [("qwen", "prompt"), ("qwen", "ft_lock")]
cfg = [(m, o, rule, k) for m, o in QW for rule in ("af5", "afpr") for k in KS]
a = lambda m, o, stem, k: disp(acc(m, o, stem, k))                      # as printed in Table tab:general
say("gen more steps raise accuracy: N1 < N8 < N40 on raw means", "12 of 12",
    f"{sum(acc(m, o, f'{r}_N1', k) < acc(m, o, f'{r}_N8', k) < acc(m, o, f'{r}_N40', k) for m, o, r, k in cfg)} of 12")
print("   not strict:", [f"{o} {r} k={k}: {a(m, o, f'{r}_N1', k)}, {a(m, o, f'{r}_N8', k)}, {a(m, o, f'{r}_N40', k)}" for m, o, r, k in cfg
                        if not a(m, o, f'{r}_N1', k) < a(m, o, f'{r}_N8', k) < a(m, o, f'{r}_N40', k)])
say("gen GN correction raises accuracy: GN N40 > Euler N40", "12 of 12",
    f"{sum(a(m, o, f'{r}_N40', k) > a(m, o, f'{r}_N40_euler', k) for m, o, r, k in cfg)} of 12")
up = sum(a(m, o, f"{r}_N40_seq", k) > a(m, o, f"{r}_N40", k) for m, o, r, k in cfg)
dn = sum(a(m, o, f"{r}_N40_seq", k) < a(m, o, f"{r}_N40", k) for m, o, r, k in cfg)
say("gen sequential above / below linear path", "5 and 6 of 12", f"{up} and {dn} of 12")
upr = sum(acc(m, o, f"{r}_N40_seq", k) > acc(m, o, f"{r}_N40", k) for m, o, r, k in cfg)
say("gen   same on raw means (above / below)", "5 and 6 of 12", f"{upr} and {12 - upr - sum(acc(m, o, f'{r}_N40_seq', k) == acc(m, o, f'{r}_N40', k) for m, o, r, k in cfg)} of 12",
    True, "the paper counts the printed tie below as neither")
print("   printed ties:", [f"{o} {r} k={k}: seq {acc(m, o, f'{r}_N40_seq', k):.4f} linear {acc(m, o, f'{r}_N40', k):.4f}" for m, o, r, k in cfg
                         if a(m, o, f"{r}_N40_seq", k) == a(m, o, f"{r}_N40", k)])
nl = {m: len(landing(m, "prompt", "af_N40_k4")["layers"]) for m in ROOT}
land_last, land_last_all = 0, 0
for m, o in SIX:
    for f in sorted((ROOT[m] / "landing" / o).glob("af_N*_k*.npz")):
        z = np.load(f)
        if int(z["k"]) >= 4:
            land_last_all += int(z["landed"][:, -1].sum()); land_last += int(z["landed"][:, -1].sum()) if f.stem.startswith("af_N40_k") else 0
say("gen AF_full items landed at the last layer, k>=4 (af_N40)", "0", str(land_last))
say("gen   same over every AF_full config (N, scheme, path)", "0", str(land_last_all))
tot, last, lacc, considered = 0, 0, [], 0
for m, o in SIX:
    for f in glob.glob(str(ROOT[m] / "graft" / o / "*" / "*.json")):
        r = json.load(open(f)); tot += 1
        considered += len(r["sel_acc"]) == nl[m] and r["sel_acc"][-1] is not None and np.isfinite(r["sel_acc"][-1])
        if r["lhat"] == nl[m] - 1:
            last += 1; lacc.append(r["test_acc"])
say("gen grafts that consider the last layer", f"{tot}/{tot}", f"{considered}/{tot}")
say("gen grafts that pick the last layer (six locks, every config)", "32 of 3820", f"{last} of {tot}")
say("gen their test accuracy", "0.00 to 0.25", f"{f2(min(lacc))} to {f2(max(lacc))}")

# ---------------- Section 6 (Discussion) --------------------------------------------------------------------------------
print("== Discussion")
three = [("qwen", "prompt"), ("qwen", "ft_lock"), ("llama", "prompt")]
say("disc k=40 AF5 and honest graft on Qwen prompt, Qwen LoRA, Llama prompt", ">= 0.91",
    f2(min(acc(m, o, s_, 40) for m, o in three for s_ in ("af5_N40", "honestk_N0"))),
    disp(min(acc(m, o, s_, 40) for m, o in three for s_ in ("af5_N40", "honestk_N0"))) >= 0.91)
say("disc sqrt((d - qk)/d), Qwen, k=40", "0.98", f2(np.sqrt((dims["qwen"] - Q * 40) / dims["qwen"])))
say("disc AF5 against FT at k=40", "0.85 against 0.88",
    f"{f2(mean6(lambda m, o: acc(m, o, 'af5_N40', 40)))} against {f2(mean6(lambda m, o: sft(m, o, 40)))}")

# ---------------- Appendix A --------------------------------------------------------------------------------------------
print("== App. A (proofs and discrete landing identities)")
fac = {}
for sch in ("_euler", ""):
    for o in ("prompt", "ft_lock"):
        for k in ((4, 10, 40) if sch else (4, 10)):
            e8, e40 = (landing("qwen", o, f"af_N{N}{sch}_k{k}")["landing_err"] for N in (8, 40))
            r = np.median(e8, 0) / np.median(e40, 0)                     # per layer, ratio of the medians over the 80 items
            fac.setdefault(sch, []).append(np.median(r) if sch else np.median(r[13:27]))   # corrected: layers 14 to 27
say("A Euler N=8 -> 40 error factor, median over layers, 2 locks x 3 k", "5.0 to 5.9", f"{f2(min(fac['_euler']), 1)} to {f2(max(fac['_euler']), 1)}")
say("A corrected factor, layers 14-27, k=4,10", "27 to 29", f"{f2(min(fac['']), 0)} to {f2(max(fac['']), 0)}")
rk = {k: sorted({int(v) for m, o in SIX for v in landing(m, o, f"seed_k{k}")["rank_stack"][:, -1]}) for k in KS}
say("A rank at the last layer, k=4,10,40 (six locks)", "8, 14, 44", ", ".join("/".join(map(str, rk[k])) for k in KS))
rk10 = {k: sorted({int(v) for m, o in TEN for v in landing(m, o, f"seed_k{k}")["rank_stack"][:, -1]}) for k in KS}
say("A   same on the ten locked models", "8, 14, 44", ", ".join("/".join(map(str, rk10[k])) for k in KS))
lb = {}
for m, o in SIX:
    hS = caps(m, o)[0][:, -1].astype(np.float64)
    rho = np.sqrt((hS ** 2).mean(-1) + EPS[m])                           # s(h_i) of the final RMSNorm
    for k in KS:
        z = landing(m, o, f"seed_k{k}")
        gS, gH = z["gamma_S"][:, -1].astype(np.float64), z["gamma_H"][:, -1].astype(np.float64)
        lb[(m, o, k)] = [max(last_layer_pair_bound(gS[i], gS[j], gH[i], gH[j], rho[i], rho[j])
                             for a_, i in enumerate(idx) for j in idx[a_ + 1:]) for idx in z["draw_items"]]
say("A last layer: min over draws k>=4 of the largest pair bound", ">= 2.50", f2(min(min(v) for v in lb.values())),
    disp(min(min(v) for v in lb.values())) >= 2.50)
print("   per lock and k, min over draws:", {f"{m}/{o}/k{k}": f2(min(v)) for (m, o, k), v in lb.items()})
say("A k=1 AF_full items landed at the last layer, Qwen prompt / LoRA", "80/80, 80/80",
    ", ".join(f"{int(landing('qwen', o, 'af_N40_k1')['landed'][:, -1].sum())}/80" for o in ("prompt", "ft_lock")))

# ---------------- Appendix B --------------------------------------------------------------------------------------------
print("== App. B (setup details)")
say("B n_L of Qwen2.5-7B", "28", str(nl["qwen"]))
say("B tab:models n_L, d (Qwen, Llama, Mistral)", "28/3584, 32/4096, 32/4096", ", ".join(f"{nl[m]}/{dims[m]}" for m in ROOT))
say("B qk at k=4,10,40", "16, 40, 160", ", ".join(str(Q * k) for k in KS))
say("B fibre dimension d - qk, Qwen", "3568, 3544, 3424", ", ".join(str(dims["qwen"] - Q * k) for k in KS))
assert dims["llama"] == dims["mistral"]
say("B fibre dimension d - qk, Llama and Mistral", "4080, 4056, 3936", ", ".join(str(dims["llama"] - Q * k) for k in KS))
st = {(float(z["target_margin"]), float(z["tol"]), float(z["rtol"]), int(z["N"])) for m, o in TEN
      for z in (landing(m, o, f"{s_}_k{k}") for s_ in RULES for k in KS)}
say("B target margin, landing tol, floor rtol, N (main configs)", "5, 0.5, 0.0001, 40", "; ".join(", ".join(f"{v:g}" for v in s_) for s_ in st))
k1 = sorted(f"{m}/{o}" for m, o in TEN if list((ROOT[m] / "landing" / o).glob("*_k1.npz")))
say("B k=1 only on Qwen prompt and Qwen LoRA seed 1", "qwen/ft_lock, qwen/prompt", ", ".join(k1))
cw = {s_: [] for s_ in RULES}
for m, o in TEN:
    mS = caps(m, o)[0].mean(0)
    for s_ in RULES:
        z = landing(m, o, f"{s_}_k40"); hs, dl = manufactured(z, caps(m, o)[0]), z["delta"]
        for r in recs(m, o, s_, 40):
            l, it = r["lhat"], np.asarray(r["items"])
            assert list(z["draw_items"][r["draw"]]) == r["items"]
            cw[s_].append(float(unit(hs[it, l].mean(0) - mS[l]) @ unit(dl[r["draw"], l])))
say("B cos(w_lhat, x_N) at k=40, every draw of the 10 locks (3 rules)", ">= 0.98", f2(min(min(v) for v in cw.values()), 3),
    min(min(v) for v in cw.values()) >= 0.98)
print("   min per rule:", {s_: f2(min(v), 3) for s_, v in cw.items()}, "draws", sum(len(v) for v in cw.values()))
nfull = ntot = 0; rmin = np.inf
for m, o in TEN:
    for f in sorted((ROOT[m] / "landing" / o).glob("seed_k*.npz")):
        z = np.load(f); k = int(z["k"]); rs = z["rank_stack"][:, :-1]
        ntot += rs.size; nfull += int((rs == Q * k).sum())
        rmin = min(rmin, float((z["sigma_stack"][:, :-1, Q * k - 1] / z["sigma_stack"][:, :-1, 0]).min()))
say("B J(0) has qk singular values above 1e-4 s1, every run below last layer", f"{ntot}/{ntot}", f"{nfull}/{ntot}")
print(f"   smallest sigma_qk / sigma_1 over those (draw, layer) cells: {rmin:.1e}")
# the floor along the path: AF_full keeps m < qk directions on a moving step (n = 0..N-1) at a layer below the last
B = {c: dict(runs=0, bit=0, layers=set()) for c in ("main", "other")}; hit = draws_ = 0; lastpick = []
m5, m5last, mmin = set(), set(), np.inf
for m, o in TEN:
    for f in sorted((ROOT[m] / "landing" / o).glob("af*_N*_k*.npz")):
        z = np.load(f); k, N = int(z["k"]), int(z["N"]); lm = z["logged_m"]
        if f.stem.startswith("af5_"):
            m5 |= set(np.unique(lm[:, :-1, :N]).astype(int).tolist()); m5last |= set(np.unique(lm[:, -1, :N]).astype(int).tolist())
        elif not f.stem.startswith("af_"): mmin = min(mmin, float(lm[:, :, :N].min()))
        if not f.stem.startswith("af_N"):
            continue
        bite = (lm[:, :-1, :N] < Q * k).any(-1)                           # [draws, nL - 1]
        c = B["main" if f.stem.startswith("af_N40_k") else "other"]
        c["runs"] += 1; c["bit"] += bool(bite.any()); c["layers"] |= set((np.nonzero(bite.any(0))[0] + 1).tolist())
        c.setdefault("n", {})[f"{m}/{o}/{f.stem}"] = int(bite.any(0).sum())      # layers with a bite in this run
        for r in recs(m, o, f.stem.rsplit("_k", 1)[0], k):
            draws_ += 1
            if r["lhat"] is None: continue
            if r["lhat"] < lm.shape[1] - 1: hit += bool(bite[r["draw"], r["lhat"]])
            else: lastpick.append(f"{m}/{o}/{f.stem}/d{r['draw']}")
say("B floor bites AF_full, GN N=40 linear: runs", "8 of 32 runs", f"{B['main']['bit']} of {B['main']['runs']} runs")
say("B   at the layers (1-based)", "1, 2, 3, 31", ", ".join(map(str, sorted(B["main"]["layers"]))))
say("B floor bites AF_full, other configs (Qwen): runs", "11 of 34 runs", f"{B['other']['bit']} of {B['other']['runs']} runs")
n_seq40 = max(v for r, v in B["other"]["n"].items() if r.endswith("_seq_k40"))
n_rest = max(v for r, v in B["other"]["n"].items() if not r.endswith("_seq_k40"))
say("B   layers per run: sequential k=40 / every other run", "16 / 4", f"{n_seq40} / {n_rest}")
say("B   at the chosen layer lhat (every draw of every AF_full run)", "never", f"{hit} of {draws_} draws", hit == 0)
say("B   AF_full draws that pick the last layer (floor bites there, k>=2)", "(not stated)", f"{len(lastpick)} {lastpick[:4]}", True)
say("B m of AF5 on the steps, below / at the last layer", "5", "/".join(map(str, sorted(m5))) + " ; " + "/".join(map(str, sorted(m5last))),
    m5 == {5})
say("B m >= 1 for the truncated rules AF_PR and AF_0.1", ">= 1", f"{mmin:g}", mmin >= 1)
say("B AF records per run (N=40): Jacobian evaluations", "41", str(landing("qwen", "prompt", "af_N40_k4")["logged_m"].shape[-1]))
gmax = max(int(landing(m, o, f"gd_N1000_k{k}")["steps"].max()) for m, o in SIX for k in KS)
say("B GD stops after at most 1000 steps", "1000", str(gmax))
rc = {(json.dumps(d["recipe"], sort_keys=True)) for m, o in SIX for k in KS for d in [json.load(open(ROOT[m] / "sft" / o / f"sft_k{k}.json"))]}
lr = {m: sorted({json.load(open(ROOT[m] / "sft" / o / f"sft_k{k}.json"))["lr"] for o in ("prompt", "ft_lock") for k in KS}) for m in ROOT}
say("B FT learning rate (Qwen, Llama, Mistral)", "1e-4, 1e-4, 5e-5", ", ".join("/".join(f"{v:.0e}".replace("e-0", "e-") for v in lr[m]) for m in ROOT))
r0 = json.loads(sorted(rc)[0])
say("B FT LoRA r, alpha, dropout, targets", "16, 32, 0.05, q,k,v,o", ", ".join(str(r0["lora"][x]) for x in ("r", "alpha", "dropout", "targets")))
say("B FT examples, batch, checkpoints, clip", "1024, 4, [256, 512, 1024], 1",
    f"{r0['examples']}, {r0['train_batch']}, {r0['checkpoints']}, {r0['clip']:g}", len({json.dumps({x: v for x, v in json.loads(s_).items() if x != 'lr'}) for s_ in rc}) == 1
    and f"{r0['examples']}, {r0['train_batch']}, {r0['checkpoints']}, {r0['clip']:g}" == "1024, 4, [256, 512, 1024], 1")
spread = {key: max(max(v) - min(v) for m, o in SIX for k in KS
                   for v in [[sft(m, o, k, key, ck) for ck in ("256", "512", "1024")]]) for key in ("test", "test_obqa")}
say("B FT checkpoints differ by at most (ARC)", "0.015", f"{spread['test']:.3f}", spread["test"] <= 0.015 + 1e-9)
say("B FT checkpoints differ by at most (OpenBookQA)", "0.02", f"{spread['test_obqa']:.3f}", spread["test_obqa"] <= 0.02 + 1e-9)
