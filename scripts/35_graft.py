"""Graft every landing on the two test sets: pool-form reference per draw, layer selection on the draw's own k
labels, test at l-hat for every draw (ARC and OBQA), test curves over every layer for the curve draws. Head-shared batched tails.
A non-finite graft (a diverged landing) is recorded, never scored: its layers are listed in nonfinite_layers (sel_acc and
sel_margin NaN there, never chosen as l-hat), and a test set whose logits at l-hat are not finite gets null scores (nonfinite: true).

A landing {stem}_k{k}.npz is grafted at the draws of size k' (--draw-k):
  k' = k          the joint entry of that landing at k                     -> graft/{organism}/{stem}_k{k}/k{k}_d{d}.json
  k' > k = 1      the "single vs stacked" rows: the 80 one-item shifts averaged over each draw of size k'
                                                                            -> graft/{organism}/{stem}_k1/k{k'}_d{d}.json
Default --draw-k: [k] for k > 1. For k = 1: [1], plus every budget when --item-rows is 'all', or (default 'final')
when the landing is a corrected straight-line run at the largest N of splits.N or the final GD state."""
import time
import numpy as np, torch
from _common import *
from actflow.readout import encode, capture
from actflow.graft import grafted_logits_rows
from actflow.reference import all_draws, curve_draws, reference, manufactured
from actflow.metrics import accuracy, balanced_accuracy, letter_hist, gold_margin
from actflow.plan import item_rows, expected_grafts

ap = parser(__doc__)
ap.add_argument("--landings", nargs="+", default=None, help="landing names, e.g. af_N40_k10 gd_N1000_k4 (default: the planned ones on disk, see below)")
ap.add_argument("--draw-k", nargs="+", type=int, default=None, help="draw sizes to graft at (default: see above)")
ap.add_argument("--item-rows", default="final", choices=["none", "final", "all"], help="which k = 1 landings are also grafted at k' > 1")
ap.add_argument("--batch", type=int, default=16, help="tail rows per forward")
ap.add_argument("--chunk", type=int, default=16, help="items whose block outputs are cached at once")
ap.add_argument("--no-curves", action="store_true")
ap.add_argument("--no-obqa", action="store_true")
args = ap.parse_args()
cfg, splits, org, lids = setup(args)
nL, d, dev = cfg["model"]["n_layers"], cfg["model"]["d"], args.device

pool = items(splits, "pool")
cap_p = np.load(results_dir("captures", args.organism) / "pool.npz")
gold_p = gold_of(pool)
assert (cap_p["gold"] == gold_p).all()
TESTS = [("test", "")] + ([] if args.no_obqa else [("test_obqa", "obqa_")])   # (splits key, record-key prefix)
tests = {}
for which, pre in TESTS:
    its = items(splits, which); cap_t = np.load(results_dir("captures", args.organism) / f"{which}.npz"); g = gold_of(its)
    assert (cap_t["gold"] == g).all()
    tests[which] = dict(items=its, prompts=prompts_of(org, its), gold=g, acc_locked=float(cap_t["acc_locked"]), pre=pre)
honest = load_json(results_dir("captures", args.organism) / "honest.json") if (results_dir("captures", args.organism) / "honest.json").exists() else {}
h_pool_mean = torch.as_tensor(cap_p["h"].mean(0), device=dev)            # [nL, d]
prompts_p = prompts_of(org, pool)
all_d = all_draws(splits, len(pool))
curves = set() if args.no_curves else curve_draws(splits)
ldir = results_dir("landing", args.organism)
# default: the landings the blocks of this organism plan that are on disk
landings = args.landings or sorted(nm for nm in expected_grafts(args.organism, blocks_cfg(args), splits) if (ldir / f"{nm}.npz").exists())
RUN_STAMP = f"{meta()['time']} {meta().get('gpu', 'cpu')} transformers {meta().get('transformers')}"
log(f"landings: {landings}")


def cached_chunk(prompts, s, e):
    ids, mask, pos, _ = encode(org.tok, prompts[s:e], dev)
    store, _ = capture(org.model, ids, mask, pos, lids, list(range(nL)), last_only=False)
    return store, mask, pos, ids


def lhat_of(sel_acc, sel_margin):
    """argmax accuracy; ties by mean margin of the correct letter; then the smaller layer. Chosen on a masked copy:
    NaN (a non-finite layer) counts as -inf in both, so it is never chosen; None when no layer is finite."""
    acc = np.where(np.isnan(sel_acc), -np.inf, sel_acc); mg = np.where(np.isnan(sel_margin), -np.inf, sel_margin)
    if np.isneginf(acc).all():
        return None
    best = np.flatnonzero(acc == acc.max())
    best = best[np.flatnonzero(mg[best] == mg[best].max())]
    return int(best.min())


def item_rows_for(pl):
    return item_rows(landing_name(pl["rule"], pl["N"], pl["k"], pl["scheme"], pl["path"]), args.item_rows, splits)


for name in landings:
    pl = parse_landing(name); kland, stem = pl["k"], pl["stem"]
    land = np.load(ldir / f"{name}.npz")
    assert str(land["stem"]) == stem and int(land["k"]) == kland, f"{name}: file metadata disagree with its name"
    dks = args.draw_k or item_rows_for(pl)
    for dk in dks:
        assert dk == kland or kland == 1, f"{name}: only a k = 1 landing is grafted at other draw sizes (asked k'={dk} for k={kland})"
    saved = land["draw_items"]                                             # the draws the landing was manufactured on
    assert all((saved[dn] == all_d[(kland, dn)]).all() for dn in range(saved.shape[0])), \
        f"{name}: its saved draws differ from the draws of the current splits (perm_seed / pool changed?)"
    keys = [kd for kd in all_d if kd[0] in dks]
    item_draws = {i: [kd for kd in keys if i in set(all_d[kd].tolist())] for i in range(len(pool))}
    out_dir = results_dir("graft", args.organism, name)
    log(f"{name}: {stem_label(stem)} k={kland}; grafting at draw sizes {dks} ({len(keys)} draws); curve draws {sorted(curves & set(keys))}")
    if "finite" in land.files and not land["finite"].all():
        log(f"{name}: h_star not finite at {int((~land['finite']).sum())} (item, layer) cells, layers {np.flatnonzero(~land['finite'].all(0)).tolist()}")
    h_star = torch.as_tensor(manufactured(land, cap_p["h"]), device=dev)  # [n_pool, nL, d]
    refs = {kd: reference(h_star, h_pool_mean, all_d[kd]) for kd in keys}  # (u [nL,d], t [nL])
    t0 = time.time()

    # ---- selection: graft each draw's own k' items at every layer ----
    sel_logits = {kd: np.zeros((nL, len(all_d[kd]), 4), np.float32) for kd in keys}
    pos_in = {kd: {int(i): j for j, i in enumerate(all_d[kd])} for kd in keys}
    for s in range(0, len(pool), args.chunk):
        e = min(len(pool), s + args.chunk)
        store, mask, pos, ids = cached_chunk(prompts_p, s, e)
        rows = [(b, kd) for b in range(e - s) for kd in item_draws[s + b]]
        if not rows:
            continue
        ri = [b for b, _ in rows]
        for l in range(nL):
            U = torch.stack([refs[kd][0][l] for _, kd in rows]); T = torch.stack([refs[kd][1][l] for _, kd in rows])
            lg = grafted_logits_rows(org.model, store[l], l, mask, pos, lids, ri, U, T, args.batch, args.impl, ids=ids).cpu().numpy()
            for r, (b, kd) in enumerate(rows):
                sel_logits[kd][l, pos_in[kd][s + b]] = lg[r]
        del store
    sel = {}
    for kd in keys:
        g = gold_p[all_d[kd]]
        ref_ok = (torch.isfinite(refs[kd][0]).all(-1) & torch.isfinite(refs[kd][1])).cpu().numpy()      # [nL]
        bad = [l for l in range(nL) if not (ref_ok[l] and np.isfinite(sel_logits[kd][l]).all())]           # the draw's non-finite layers
        acc_l = np.array([np.nan if l in bad else accuracy(sel_logits[kd][l].argmax(-1), g) for l in range(nL)])   # NaN kept in the record
        mg_l = np.array([np.nan if l in bad else gold_margin(sel_logits[kd][l], g).mean() for l in range(nL)])
        sel[kd] = (acc_l, mg_l, lhat_of(acc_l, mg_l), bad)
    nbad = sum(len(sel[kd][3]) > 0 for kd in keys); nnone = sum(sel[kd][2] is None for kd in keys)
    log(f"{name}: selection done {time.time()-t0:.0f}s; l-hat hist {np.bincount(np.array([sel[kd][2] for kd in keys if sel[kd][2] is not None], int), minlength=nL).tolist()}"
        + (f"; {nbad} draws with non-finite layers, {nnone} with no finite layer (l-hat null)" if nbad else ""))

    # ---- test: every draw at its l-hat, curve draws at all layers, on each test set ----
    need = {kd: (set(range(nL)) if kd in curves else ({sel[kd][2]} if sel[kd][2] is not None else set())) for kd in keys}
    by_layer = {l: [kd for kd in keys if l in need[kd]] for l in range(nL)}
    TL = {}
    for which, tinfo in tests.items():
        test_logits = {kd: {l: np.zeros((len(tinfo["items"]), 4), np.float32) for l in need[kd]} for kd in keys}
        for s in range(0, len(tinfo["items"]), args.chunk):
            e = min(len(tinfo["items"]), s + args.chunk)
            store, mask, pos, ids = cached_chunk(tinfo["prompts"], s, e)
            for l in range(nL):
                kds = by_layer[l]
                if not kds:
                    continue
                rows = [(b, kd) for kd in kds for b in range(e - s)]
                ri = [b for b, _ in rows]
                U = torch.stack([refs[kd][0][l] for _, kd in rows]); T = torch.stack([refs[kd][1][l] for _, kd in rows])
                lg = grafted_logits_rows(org.model, store[l], l, mask, pos, lids, ri, U, T, args.batch, args.impl, ids=ids).cpu().numpy()
                for r, (b, kd) in enumerate(rows):
                    test_logits[kd][l][s + b] = lg[r]
            del store
        TL[which] = test_logits
    got = {}                                                               # (test set, draw) -> test acc at l-hat, None if non-finite
    for kd in keys:
        k, dn = kd
        acc_l, mg_l, lhat, bad = sel[kd]
        rec = {"rule": pl["rule"], "N": pl["N"], "scheme": pl["scheme"], "path": pl["path"], "rank_spec": str(land["rank_spec"]),
               "stem": stem, "landing": name, "k_landing": kland, "k": k, "draw": dn, "items": all_d[kd].tolist(),
               "sel_acc": acc_l.tolist(), "sel_margin": mg_l.tolist(), "lhat": lhat, "nonfinite_layers": bad, "nonfinite": False,
               "provenance": f"35_graft.py {RUN_STAMP} from landing {name} ({str(land['provenance']) if 'provenance' in land.files else 'no landing provenance'})"}
        for which, tinfo in tests.items():
            pre, gold_t = tinfo["pre"], tinfo["gold"]
            lg_ = TL[which][kd][lhat] if lhat is not None else None
            if lg_ is not None and np.isfinite(lg_).all():
                ans = lg_.argmax(-1)
                rec.update({f"{pre}test_acc": accuracy(ans, gold_t), f"{pre}test_answers": ans.tolist(), f"{pre}test_correct": (ans == gold_t).astype(int).tolist(),
                            f"{pre}letter_hist": letter_hist(ans), f"{pre}balanced_acc": balanced_accuracy(ans, gold_t),
                            f"{pre}test_margin": float(gold_margin(lg_, gold_t).mean())})
            else:                                                          # argmax of NaN logits is letter A: never score it
                rec.update({f"{pre}{key}": None for key in ("test_acc", "test_answers", "test_correct", "letter_hist", "balanced_acc", "test_margin")})
                rec["nonfinite"] = True
            rec[f"{pre}acc_locked"] = tinfo["acc_locked"]
            got[which, kd] = rec[f"{pre}test_acc"]
            if which in honest:
                rec[f"{pre}acc_honest"] = honest[which]["acc_honest"]
            if kd in curves:
                ok = [l not in bad and bool(np.isfinite(TL[which][kd][l]).all()) for l in range(nL)]
                curve = [accuracy(TL[which][kd][l].argmax(-1), gold_t) if ok[l] else None for l in range(nL)]
                lopt = int(np.nanargmax(np.array(curve, float))) if any(ok) else None      # NaN-aware: first max over finite layers
                rec[f"{pre}test_acc_all"] = curve; rec[f"{pre}lopt"] = lopt; rec[f"{pre}test_acc_at_lopt"] = curve[lopt] if lopt is not None else None
                rec[f"{pre}letter_hist_all"] = [letter_hist(TL[which][kd][l].argmax(-1)) if ok[l] else None for l in range(nL)]
        save_json(out_dir / f"k{k}_d{dn}.json", rec)
    for k in dks:
        for which in tests:
            accs = [got[which, kd] for kd in keys if kd[0] == k]; fin = [a for a in accs if a is not None]
            msg = f"acc at l-hat mean {np.mean(fin):.3f} (n={len(fin)}, min {min(fin):.2f}, max {max(fin):.2f})" if fin else "no finite draw"
            log(f"{name} at k'={k:2d} on {which}: {msg}" + (f"; {len(accs) - len(fin)} non-finite draws skipped" if len(fin) < len(accs) else ""))
    log(f"{name}: done {(time.time()-t0)/60:.1f} min peak={cuda_peak():.1f}GB")
save_json(results_dir("graft", args.organism) / "meta.json", meta(args))
