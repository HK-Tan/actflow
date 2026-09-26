"""Manufacture the landings of one block (configs/blocks.yaml) or of an explicit
rule x N x scheme x path set, on the draws of every size k, at every layer, with the per-step record.
One code path: the k items of a draw are lifted TOGETHER by one common shift delta (stacked Jacobian J_K [4k, d]);
k = 1 is the per-item rule. Batches are ordered by the pool permutation so every draw is a contiguous block in one batch.
-> results_{cfg}/landing/{organism}/{rule}_N{N}[_euler][_seq]_k{k}.npz      one file per run (io.stem_of): the shift delta of each
                                                                     draw at every layer (h* = h_S + delta, reference.manufactured),
                                                                     landing numbers and the per-step record
   results_{cfg}/landing/{organism}/gd_N{max_steps}_k{k}.npz               GD, the same
   results_{cfg}/landing/{organism}/seed_k{k}.npz                          gamma_S, gamma_H, spectrum and top right vectors of J_K at t = 0
   results_{cfg}/landing/{organism}/meta_{tag}.json
Examples:  --block A1 --k 1        --block A1 --rules gd --k 10 40        --block A2        --rules af --N 40 --scheme euler --k 40"""
import time
import numpy as np, torch
from _common import *
from actflow.readout import OneTokenTail
from actflow.data import permutation
from actflow.plan import runs_for as plan_runs_for, graft_names as plan_graft_names
from actflow.landing import (raise_to, line_points, seq_points, seq_orders, pinv_step, landing_numbers, stacked_gram_stats,
                             stacked_spectrum, actflow, gd, REC_KEYS, GD_KEYS)

ap = parser(__doc__)
ap.add_argument("--block", default=None, help="A1 | A2 | A3 | B | C (configs/blocks.yaml); --rules/--N/--k/--scheme/--path narrow it")
ap.add_argument("--rules", nargs="+", default=None, help="rule names of splits.rules (default: the block's, else all)")
ap.add_argument("--N", nargs="+", type=int, default=None, help="flow steps (default: the block's, else splits.N)")
ap.add_argument("--scheme", default=None, choices=["corrected", "euler"])
ap.add_argument("--path", default=None, choices=["line", "seq"])
ap.add_argument("--k", nargs="+", type=int, default=None, help="draw sizes (default: the block's, else splits.budgets)")
ap.add_argument("--batch", type=int, default=40, help="pool rows per one-token tail (40 fits a 48 GB card at Qwen 7B fp32; 00_check measures the peak)")
ap.add_argument("--margin", type=float, default=None)
ap.add_argument("--rtol", type=float, default=None, help="always refused: no stem records it, and it would also move seed_k{k}.npz's "
                "floor (AF_0.1 is rules.af01.rtol in configs/splits.yaml; the floor of every other rule is splits.rtol)")
ap.add_argument("--no-spectrum", action="store_true", help="skip the t = 0 spectrum and top right vectors in seed_k{k}.npz")
ap.add_argument("--tag", default=None, help="suffix of meta_{tag}.json (default: block or 'runs')")
args = ap.parse_args()
cfg, splits, org, lids = setup(args)
nL, d, dev = cfg["model"]["n_layers"], cfg["model"]["d"], args.device
m = args.margin if args.margin is not None else float(splits["margin"])
assert args.rtol is None, (f"--rtol {args.rtol}: no stem records it, so the run would be saved under another rule's names; "
                           f"AF_0.1 is rules.af01.rtol in configs/splits.yaml (--rules af01): never pass --rtol")
rtol = float(splits["rtol"])
RULES = splits["rules"]
GD = splits["gd"]; gd_tol, gd_steps, gd_first = float(GD["tol"]), int(GD["max_steps"]), float(GD["first_step"])
vtop = int(splits.get("seed", {}).get("vtop", 20))
order_seed = int(splits.get("order_seed", 0))

if args.block:
    spec = blocks_cfg(args)[args.block]
    assert args.organism in spec["organisms"], f"block {args.block} has no cell {args.organism}"
    rules = args.rules or list(spec["rules"]); Ns = args.N or list(spec["N"])
    scheme = args.scheme or spec["scheme"]; path = args.path or spec["path"]; ks = args.k or list(spec["k"])
else:
    rules = args.rules or list(RULES); Ns = args.N or list(splits["N"])
    scheme = args.scheme or "corrected"; path = args.path or "line"; ks = args.k or list(splits["budgets"])
for r in rules:
    assert r in RULES, f"unknown rule {r}; splits.rules = {list(RULES)}"
tag = args.tag or args.block or "runs"

its = items(splits, "pool"); prompts = prompts_of(org, its); gold_np = gold_of(its); n = len(its)
cap = np.load(results_dir("captures", args.organism) / "pool.npz")
assert (cap["gold"] == gold_np).all() and list(cap["letter_ids"]) == lids
perm = permutation(n, splits["perm_seed"])
batch = min(args.batch, n)
for k in ks:
    assert batch % k == 0 and n % k == 0, f"k={k} must divide batch={batch} and pool={n} (draws are contiguous blocks)"


def runs_for(k):
    """[(stem, rule, N)] at this k (actflow.plan.runs_for: the expansion 40_analyse.py checks against)."""
    return plan_runs_for(k, rules, Ns, scheme, path, RULES, gd_steps)


def gd_names():
    return [stem_of("gd", gd_steps)]


def graft_names(k):
    """The landings of this k that run_A.sh grafts: every AF landing and GD's final state only (actflow.plan)."""
    return plan_graft_names(runs_for(k), gd_steps)


RUN_STAMP = f"30_manufacture.py {meta()['time']} {meta().get('gpu', 'cpu')} transformers {meta().get('transformers')} argv {' '.join(meta()['argv'])}"
log(f"block={args.block} rules={rules} N={Ns} scheme={scheme} path={path} k={ks} m={m} rtol={rtol} batch={batch} gd={GD}")
out_dir = results_dir("landing", args.organism)
layers = list(range(nL))
for k in ks:
    D = n // k
    runs_k = runs_for(k)
    if not runs_k:
        log(f"k={k}: nothing to run"); continue
    names = [nm for nm, rule, _ in runs_k if rule != "gd"] + (gd_names() if any(r == "gd" for _, r, _ in runs_k) else [])
    draw_items = np.stack([perm[dd * k:(dd + 1) * k] for dd in range(D)])
    orders = seq_orders(k, range(D), order_seed) if path == "seq" else None
    nsteps = {nm: (N if rule != "gd" else gd_steps) for nm, rule, N in runs_k}

    def fresh(nm):
        a = {"F": np.zeros((n, nL, 4), np.float32),
             "landing_err": np.zeros((n, nL), np.float32), "margin": np.zeros((n, nL), np.float32),
             "landed": np.zeros((n, nL), bool), "gold_first": np.zeros((n, nL), bool), "steps": np.zeros((n, nL), np.int32),
             "shift": np.zeros((n, nL), np.float32), "seconds": np.zeros(nL, np.float32),
             "delta": np.zeros((D, nL, d), np.float32), "delta_norm": np.zeros((D, nL), np.float32),
             "delta_rel": np.zeros((D, nL), np.float32), "cos_vs_mean_oneshot": np.zeros((D, nL), np.float32),
             "norm_vs_mean_oneshot": np.zeros((D, nL), np.float32), "norm_vs_item_oneshot": np.zeros((D, nL), np.float32),
             "all_landed": np.zeros((D, nL), bool)}
        if nm.startswith("gd_"):
            if nm == stem_of("gd", gd_steps):
                for key in GD_KEYS:
                    a[f"logged_{key}"] = np.full((D, nL, gd_steps + 1), np.nan, np.float32)
        else:
            N = nsteps[nm]
            for key in REC_KEYS:
                a[f"logged_{key}"] = np.full((D, nL, N + 1), np.nan, np.float32)
            a["logged_sigma"] = np.full((D, nL, N + 1, 4 * k), np.nan, np.float32)
        return a
    acc = {nm: fresh(nm) for nm in names}
    seed = {"gamma_S": np.zeros((n, nL, 4), np.float32), "gamma_H": np.zeros((n, nL, 4), np.float32),
            "rank_stack": np.zeros((D, nL), np.int32), "sigma_ratio_stack": np.zeros((D, nL), np.float32)}
    vt = min(vtop, 4 * k)
    if not args.no_spectrum:
        seed["sigma_stack"] = np.zeros((D, nL, 4 * k), np.float32)         # full singular spectrum of J_K at t = 0, descending
        seed["vtop"] = np.zeros((D, nL, vt, d), np.float16)                 # top right singular directions v_a at t = 0: unit where valid, else 0
        seed["vtop_valid"] = np.zeros((D, nL, vt), bool)                    # sigma_a > rtol * sigma_1 (rank J_K <= 4 + k at the last layer)
    tk = time.time()
    for s in range(0, n, batch):
        order = perm[s:s + batch]                                  # pool indices of this batch, draw-contiguous
        B = len(order)
        groups = [torch.arange(j * k, (j + 1) * k, device=dev) for j in range(B // k)]
        dids = [s // k + j for j in range(B // k)]                 # draw ids of the groups
        t0 = time.time()
        tail = OneTokenTail(org.model, org.tok, [prompts[i] for i in order], lids, dev, impl=args.impl)
        dlog = float(np.abs(tail.logits_S.cpu().numpy() - cap["logits"][order]).max())
        log(f"k={k} batch {s}-{s+B} ({B // k} draws): prefill {time.time()-t0:.1f}s  |logits - captures|={dlog:.2e}  peak={cuda_peak():.1f}GB")
        tgt = torch.as_tensor(gold_np[order], device=dev)
        for l in layers:
            t1 = time.time()
            h0 = tail.h_S[:, l]
            F0, J0 = tail.FJ(h0, l)
            gH = raise_to(F0, tgt, m)
            seed["gamma_S"][order, l] = F0.cpu().numpy(); seed["gamma_H"][order, l] = gH.cpu().numpy()
            rk, sr = stacked_gram_stats(J0, groups, rtol)
            seed["rank_stack"][dids, l] = rk; seed["sigma_ratio_stack"][dids, l] = sr
            if not args.no_spectrum:
                sigs, vts, oks = stacked_spectrum(J0, groups, top=vt, rtol=rtol)     # null rows zeroed before the fp16 cast
                for gi, sg, v_, ok in zip(dids, sigs, vts, oks):
                    seed["sigma_stack"][gi, l] = sg.cpu().numpy(); seed["vtop"][gi, l] = v_.cpu().numpy().astype(np.float16)
                    seed["vtop_valid"][gi, l] = ok.cpu().numpy()
            d_item = pinv_step(J0, gH - F0)                        # per-item one shots, for the pooled comparison

            def record(nm, h, steps, t2):
                F = tail.F(h, l)
                err, mg, landed, gfirst = landing_numbers(F, gH, tgt, tol=gd_tol)
                a = acc[nm]
                a["F"][order, l] = F.cpu().numpy()
                a["landing_err"][order, l] = err.cpu().numpy(); a["margin"][order, l] = mg.cpu().numpy()
                a["landed"][order, l] = landed.cpu().numpy(); a["gold_first"][order, l] = gfirst.cpu().numpy()
                a["steps"][order, l] = steps.cpu().numpy(); a["shift"][order, l] = (F - F0).mean(-1).cpu().numpy()
                for gi, idx in zip(dids, groups):
                    delta = (h - h0)[idx[0]]                        # common step of the draw
                    dm = d_item[idx].mean(0)
                    a["delta"][gi, l] = delta.cpu().numpy(); a["delta_norm"][gi, l] = float(delta.norm())
                    a["delta_rel"][gi, l] = float(delta.norm() / h0[idx].norm(dim=-1).mean())      # |delta| / mean_i |h_S,i|
                    a["cos_vs_mean_oneshot"][gi, l] = float((delta @ dm) / (delta.norm() * dm.norm() + 1e-12))
                    a["norm_vs_mean_oneshot"][gi, l] = float(delta.norm() / (dm.norm() + 1e-12))
                    a["norm_vs_item_oneshot"][gi, l] = float(delta.norm() / (d_item[idx].norm(dim=-1).mean() + 1e-12))
                    a["all_landed"][gi, l] = bool(landed[idx].all())
                a["seconds"][l] += time.time() - t2

            for nm, rule, N in runs_k:
                t2 = time.time()
                if rule == "gd":
                    grec = {}
                    h, steps, _ = gd(tail, h0, l, gH, groups, tol=gd_tol, max_steps=gd_steps, first_step=gd_first, logged=grec)
                    fin = stem_of("gd", gd_steps)
                    for gi_, gi in enumerate(dids):
                        for key in GD_KEYS:
                            v_ = np.asarray(grec[gi_][key], np.float32); acc[fin][f"logged_{key}"][gi, l, :len(v_)] = v_
                    record(fin, h, steps, t2)
                else:
                    spec_r = RULES[rule]
                    pts = line_points(F0, gH, N) if path == "line" else seq_points(F0, gH, N, groups, [orders[gi] for gi in dids])
                    frec = {}
                    h = actflow(tail, h0, l, pts, groups, rtol=float(spec_r.get("rtol", rtol)), rank=spec_r.get("rank"),
                                correction=(scheme == "corrected"), logged=frec, gold=tgt)
                    a = acc[nm]
                    for gi_, gi in enumerate(dids):
                        for key in REC_KEYS:
                            a[f"logged_{key}"][gi, l] = np.asarray(frec[gi_][key], np.float32)
                        a["logged_sigma"][gi, l] = torch.stack(frec[gi_]["sigma"]).numpy()
                    record(nm, h, torch.full((B,), N, dtype=torch.long, device=dev), t2)
            msg = "  ".join(f"{nm}:{acc[nm]['landed'][order, l].mean():.2f}/{acc[nm]['landing_err'][order, l].mean():.2f}"
                            f"/|d|{acc[nm]['delta_norm'][dids, l].mean():.1f}" for nm in names)
            log(f"  k={k} l={l:2d} rank_stack(rtol)={min(rk)}..{max(rk)}/{4*k} sr_min={min(sr):.1e}  landed/err/|delta|  {msg}  {time.time()-t1:.0f}s")
        del tail; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    common = dict(k=np.int64(k), draw_items=draw_items, layers=np.array(layers), target_margin=np.float32(m),
                  tol=np.float32(gd_tol), rtol=np.float32(rtol), gold=gold_np, provenance=RUN_STAMP)
    for nm in names:
        a = acc[nm]
        pl = parse_landing(f"{nm}_k{k}")
        spec_r = RULES[pl["rule"]]
        rk_ = spec_r.get("rank")
        fin = np.ones((n, nL), bool)                                       # [n, nL]: False where the walk diverged (35_graft records it)
        for dd, it in enumerate(draw_items):
            fin[it] = np.isfinite(a["delta"][dd]).all(-1)[None]
        save_npz(out_dir / f"{nm}_k{k}.npz", **a, finite=fin, stem=nm, rule=pl["rule"], N=np.int64(pl["N"]), scheme=pl["scheme"], path=pl["path"],
                 rank_spec=("full" if rk_ is None else str(rk_)), rule_rtol=np.float32(spec_r.get("rtol", rtol)),
                 order=(np.stack(orders) if orders is not None else np.zeros((0,), np.int64)), **common)
        lr = a["landed"].astype(float).mean(0)
        log(f"{nm}_k{k}: landed rate per layer {np.round(lr, 2).tolist()}  err median per layer {np.round(np.median(a['landing_err'], 0), 2).tolist()}  "
            f"non-finite {int((~fin).sum())}  {a['seconds'].sum()/60:.1f} min")
    save_npz(out_dir / f"seed_k{k}.npz", **seed, **common)
    log(f"k={k} done {(time.time()-tk)/60:.1f} min peak={cuda_peak():.1f}GB")
save_json(out_dir / f"meta_{tag}.json", meta(args))
produced = [f"{nm}_k{k}" for k in ks for nm in graft_names(k)]
(out_dir / f"produced_{tag}.txt").write_text("".join(nm_ + "\n" for nm_ in produced))   # run_A.sh grafts exactly these (empty file = nothing)
log(f"done; {len(produced)} landings to graft -> produced_{tag}.txt")
