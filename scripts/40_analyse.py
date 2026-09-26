"""Offline: ladders, curves, landing numbers and per-step record summaries -> results_{cfg}/analysis/{organism}.json and
results_{cfg}/analysis/{organism}_per_step.npz.  Reads graft/{organism}/{landing}/k{k}_d{d}.json and landing/{organism}/*.npz.

  joint[stem][k]  = landing {stem}_k{k} grafted at its own draws of size k                     (the ladders)
  item[stem][k]   = landing {stem}_k1 grafted at the draws of size k ("single vs stacked" rows, only for the k = 1 landings
                    grafted at some k > 1); item[stem][1] is joint[stem][1]
  landing[name]   = per-layer medians over items / draws: landing error e_N, its kept part, steps, shift, kept count m at t = 0, 1/2, 1
  _per_step.npz   = per landing, the per-step record medians over draws, [nL, N+1] each (and GD loss / err over steps)
A non-finite (diverged) draw has {pre}test_acc null in its graft record. It enters the mean, the dots and the draw and binomial
SEs at the locked accuracy of that test set (recovery 0: the graft did nothing), and every entry reports {pre}n_nonfinite.
Every ordering (ladders, landings, curves, printed rows) is (rule order of splits.rules, N, scheme, path, k), not string order."""
import glob, warnings
import numpy as np
from _common import *
from actflow.plan import expected_grafts
from actflow.metrics import binomial_se
from actflow.reference import manufactured

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--organism", required=True)
ap.add_argument("--cfg", default="qwen7b", help="model config configs/{cfg}.yaml: qwen7b, llama8b, mistral7b (results_{cfg}/ unless qwen7b)")
ap.add_argument("--tiny", action="store_true")
ap.add_argument("--partial", action="store_true", help="analyse even if some planned landings or draw sizes have no grafts yet "
                "(the missing list is printed and saved as 'missing'); without it a missing slice stops the analysis")
args = ap.parse_args()
use_cfg(args.cfg, args.tiny)
if args.organism not in load_cfg(args.cfg)["organisms"]:                 # before results_dir(), which makes the folders
    raise SystemExit(f"unknown organism {args.organism}; configs/{args.cfg}.yaml has {list(load_cfg(args.cfg)['organisms'])}")
splits = load_cfg("splits")
if args.tiny:
    from _common import TINY_SPLITS; splits.update(TINY_SPLITS)
warnings.filterwarnings("ignore", "All-NaN slice|Mean of empty slice")   # a diverged layer is NaN on purpose; nonfinite_count names it
gdir, ldir = results_dir("graft", args.organism), results_dir("landing", args.organism)
out = {"organism": args.organism, "cfg": args.cfg, "joint": {}, "item": {}, "curves": {}, "landing": {}, "seed": {}, "provenance": {}}
PRE = ["", "obqa_"]
RULES, SCHEMES, PATHS = list(splits["rules"]), list(splits.get("schemes", ["corrected", "euler"])), list(splits.get("paths", ["line", "seq"]))


def order(name):
    """sort key of a landing name or a stem: (rule order of splits.rules, N, scheme, path, k), so N=8 sorts before N=40"""
    pl = parse_landing(name if "_k" in name else name + "_k0")
    at = lambda xs, x: xs.index(x) if x in xs else len(xs)
    return (at(RULES, pl["rule"]), pl["N"], at(SCHEMES, pl["scheme"]), at(PATHS, pl["path"]), pl["k"])


def read(f, how):
    """load one results file; a truncated or half-rsynced one stops the analysis and names the file"""
    try:
        return how(f)
    except Exception as ex:
        raise SystemExit(f"cannot read {f} ({type(ex).__name__}: {ex}): half-written, or pulled mid-rsync? "
                         "rsync the pod again after its final line, or rerun the slice that writes it")


def npz(f):
    with np.load(f) as z:
        return {k: z[k] for k in z.files}                                  # eager, so a corrupt member fails inside read()


def n_test_of(rs, pre):
    """test-set size from the first record with answers (a non-finite record has test_answers null), else splits"""
    return next((len(r[f"{pre}test_answers"]) for r in rs if r.get(f"{pre}test_answers") is not None), int(splits["test_obqa" if pre else "test"]["n"]))


def entry(rs):
    rs = sorted(rs, key=lambda r: r["draw"])
    e = {"n_draws": len(rs), "lhat": [r["lhat"] for r in rs]}
    for pre in PRE:
        if f"{pre}test_acc" not in rs[0]:
            continue
        fin = [r for r in rs if r[f"{pre}test_acc"] is not None]
        # a non-finite draw counts at its record's locked accuracy (recovery 0) in the mean, the dots and the draw / binomial SEs
        accs = np.array([r[f"{pre}test_acc"] if r[f"{pre}test_acc"] is not None else r[f"{pre}acc_locked"] for r in rs], float)
        e[f"{pre}n_nonfinite"] = len(rs) - len(fin)
        e[f"{pre}mean"] = float(accs.mean()); e[f"{pre}binomial_se"] = binomial_se(float(accs.mean()), n_test_of(rs, pre))
        e[f"{pre}draw_se"] = float(accs.std(ddof=1) / np.sqrt(len(accs))) if len(accs) > 1 else None
        if len(fin) == len(rs) and f"{pre}test_correct" in rs[0]:           # SE over the shared test items of the mean-over-draws accuracy;
            C = np.array([r[f"{pre}test_correct"] for r in rs], float)     # [draws, n_test]. None when a draw is non-finite (it has no answers): the binomial SE is used
            e[f"{pre}item_se"] = float(C.mean(0).std(ddof=1) / np.sqrt(C.shape[1]))
        else:
            e[f"{pre}item_se"] = None
        e[f"{pre}dots"] = accs.tolist()
        # letter shares and balanced accuracy: finite draws only, None when there are none
        e[f"{pre}letter_hist_mean"] = np.mean([r[f"{pre}letter_hist"] for r in fin], 0).tolist() if fin else None
        e[f"{pre}max_letter_share"] = float(np.max([max(r[f"{pre}letter_hist"]) for r in fin])) if fin else None
        e[f"{pre}balanced_acc_mean"] = float(np.mean([r[f"{pre}balanced_acc"] for r in fin])) if fin else None
    # the "train" side: the selection score, i.e. the grafted accuracy at l-hat on the draw's own k labelled items. The graft
    # was built from these items and l-hat was chosen on them, so this is optimistic; by_layer is the selection statistic,
    # not a layer profile of steering (use the held-out curves of the curve draws for anything over layers).
    # NaN-aware: sel_acc / sel_margin are NaN at a draw's non-finite layers, and a draw with lhat null has no score.
    sc = [r for r in rs if r["lhat"] is not None]
    e["train_acc_mean"] = float(np.nanmean(np.array([r["sel_acc"][r["lhat"]] for r in sc], float))) if sc else None
    e["train_margin_mean"] = float(np.nanmean(np.array([r["sel_margin"][r["lhat"]] for r in sc], float))) if sc else None
    e["train_acc_by_layer_mean"] = np.nanmean(np.array([r["sel_acc"] for r in rs], float), 0).tolist()
    if "mean" in e and e["train_acc_mean"] is not None:
        e["train_test_gap"] = e["train_acc_mean"] - e["mean"]
    return e


# ---- ladders and curves from the graft jsons ----
n_pool = int(splits["pool"]["n"])
found = {}                                                              # landing name -> draw sizes k' present
for gd_ in sorted((p for p in gdir.iterdir() if p.is_dir()), key=lambda p: order(p.name)):
    name = gd_.name
    pl = parse_landing(name); stem, kland = pl["stem"], pl["k"]
    recs = [read(f, load_json) for f in sorted(glob.glob(str(gd_ / "k*_d*.json")))]
    if not recs:
        continue
    for pre in PRE:
        if f"{pre}test_acc" in recs[0]:
            out[f"{pre}n_test"] = n_test_of(recs, pre)
        if f"{pre}acc_locked" in recs[0]:
            out[f"{pre}acc_locked"] = recs[0][f"{pre}acc_locked"]
        if f"{pre}acc_honest" in recs[0]:
            out[f"{pre}acc_honest"] = recs[0][f"{pre}acc_honest"]
    assert all(r["landing"] == name and r["k_landing"] == kland for r in recs), f"{name}: a record's metadata disagree with its directory"
    found[name] = {int(r["k"]) for r in recs}
    for k in sorted(found[name]):
        rk = [r for r in recs if r["k"] == k]
        draws_seen = sorted(r["draw"] for r in rk)
        assert draws_seen == list(range(n_pool // k)), f"{name} at k'={k}: expected draws 0..{n_pool // k - 1}, found {draws_seen}"
        out["provenance"].setdefault(name, {})[str(k)] = sorted({r.get("provenance", "none") for r in rk})
        e = entry(rk)
        if k == kland:
            out["joint"].setdefault(stem, {})[str(k)] = e
        if kland == 1 and max(found[name]) > 1:                             # single vs stacked: only k = 1 landings grafted at some k' > 1
            out["item"].setdefault(stem, {})[str(k)] = e
    out["curves"][name] = {f"k{r['k']}_d{r['draw']}": {"test_acc_all": r["test_acc_all"], "obqa_test_acc_all": r.get("obqa_test_acc_all"),
                                                       "lhat": r["lhat"], "lopt": r["lopt"], "obqa_lopt": r.get("obqa_lopt"), "sel_acc": r["sel_acc"],
                                                       "acc_at_lhat": r["test_acc"], "acc_at_lopt": r["test_acc_at_lopt"],
                                                       "obqa_acc_at_lhat": r.get("obqa_test_acc"), "obqa_acc_at_lopt": r.get("obqa_test_acc_at_lopt"),
                                                       "nonfinite_layers": r.get("nonfinite_layers", [])}         # null accuracies: non-finite
                           for r in sorted(recs, key=lambda r: (r["k"], r["draw"])) if "test_acc_all" in r}
for view in ("joint", "item"):
    out[view] = {stem: dict(sorted(v.items(), key=lambda kv: int(kv[0]))) for stem, v in sorted(out[view].items(), key=lambda kv: order(kv[0]))}

# ---- completeness: every landing the blocks of this organism plan (configs/blocks.yaml, actflow.plan) must have its
# grafts at every planned draw size; a slice or a whole pod that never arrived shows up here instead of vanishing ----
expected = expected_grafts(args.organism, blocks_cfg(args), splits)
missing = [f"{nm} at k'={kk}" for nm, kks in sorted(expected.items(), key=lambda kv: order(kv[0])) for kk in kks if kk not in found.get(nm, set())]
extra = sorted(set(found) - set(expected), key=order)
out["expected_landings"] = len(expected); out["missing"] = missing; out["not_in_plan"] = extra
if not expected:
    print(f"completeness: no block in the config lists {args.organism} for cfg {args.cfg}; nothing to check")
elif missing:
    print(f"completeness: {len(missing)} planned graft sets missing of {sum(len(v) for v in expected.values())}:")
    for x in missing:
        print("   ", x)
    if not args.partial:
        raise SystemExit("stopping: rsync the missing pods' results_{cfg}/ (or rerun their slices), or pass --partial to analyse what is there")
else:
    print(f"completeness: all {len(expected)} planned landings grafted at every planned draw size")
if extra:
    print(f"note: {len(extra)} grafted landings are not in the plan (kept in the analysis): {extra}")

# ---- landing numbers and per-step record summaries from the npz ----
cap = read(results_dir("captures", args.organism) / "pool.npz", npz)
hS = cap["h"]
def med(x, ax=0): return np.nanmedian(x, ax).tolist()
def lorder(p): return (0, (int(p.stem[len("seed_k"):]),)) if p.stem.startswith("seed_k") else (1, order(p.stem))   # seed_k{k} first, by k
per_step = {}
for p in sorted(ldir.glob("*_k*.npz"), key=lorder):
    z = read(p, npz)
    if p.stem.startswith("seed"):
        k = int(z["k"])
        rec = {"rank_stack_min": z["rank_stack"].min(0).tolist(), "sigma_ratio_stack_median": med(z["sigma_ratio_stack"]),
               "sigma_ratio_stack_min": z["sigma_ratio_stack"].min(0).tolist()}
        if "sigma_stack" in z:
            sg = np.median(z["sigma_stack"], 0)                          # [nL, 4k]
            rec["sigma_top5_median"] = sg[:, :5].tolist(); rec["sigma_last_median"] = sg[:, -1].tolist()
        out["seed"][f"k{k}"] = rec
        continue
    pl = parse_landing(p.stem)
    rec = {"rule": pl["rule"], "N": pl["N"], "scheme": pl["scheme"], "path": pl["path"], "k": pl["k"],
           "rank_spec": str(z["rank_spec"]) if "rank_spec" in z else None, "tol": float(z["tol"]),
           "landed_rate": z["landed"].astype(float).mean(0).tolist(), "all_landed_rate": z["all_landed"].astype(float).mean(0).tolist(),
           "gold_first_rate": z["gold_first"].astype(float).mean(0).tolist(), "landing_err_median": med(z["landing_err"]),
           "margin_median": med(z["margin"]), "steps_median": med(z["steps"].astype(float)), "shift_median": med(z["shift"]),
           "step_rel_median": med(np.linalg.norm(manufactured(z, hS) - hS, axis=-1) / np.linalg.norm(hS, axis=-1)),
           "delta_norm_median": med(z["delta_norm"]), "delta_rel_median": med(z["delta_rel"]),
           "cos_vs_mean_oneshot_median": med(z["cos_vs_mean_oneshot"]), "norm_vs_item_oneshot_median": med(z["norm_vs_item_oneshot"]),
           "seconds": z["seconds"].tolist(), "provenance": str(z["provenance"]) if "provenance" in z else "none"}
    if "finite" in z:
        rec["nonfinite_count"] = (~z["finite"]).sum(0).tolist()          # [nL]: pool rows whose h* is not finite (a diverged walk)
    lg = {}
    for key in [f[len("logged_"):] for f in z if f.startswith("logged_") and f != "logged_sigma"]:
        lg[key] = np.nanmedian(z[f"logged_{key}"], 0)                        # [nL, steps+1], median over draws
    if "logged_sigma" in z:
        lg["sigma"] = np.nanmedian(z["logged_sigma"], 0)                     # [nL, N+1, 4k]
    if "logged_e_kept" in z:
        N = pl["N"]
        rec["e_N_draw_median"] = lg["e"][:, N].tolist()                  # whole-draw 2-norm at t = 1 (same aggregation as the two parts)
        rec["e_kept_N_median"] = lg["e_kept"][:, N].tolist(); rec["e_dropped_N_median"] = lg["e_dropped"][:, N].tolist()
        rec["m_at_t"] = {t: lg["m"][:, int(round(t * N))].tolist() for t in (0, 0.5, 1)}
        rec["pr_at_t"] = {t: lg["pr"][:, int(round(t * N))].tolist() for t in (0, 0.5, 1)}
        rec["label_acc_N_median"] = lg["label_acc"][:, N].tolist(); rec["share0_N_median"] = lg["share0"][:, N].tolist()
    if "logged_loss" in z:
        L_ = z["logged_loss"]; last = np.sum(~np.isnan(L_), -1) - 1                        # the last recorded step per (draw, layer)
        rec["gd_loss_last_median"] = np.nanmedian(np.take_along_axis(L_, last[..., None].clip(min=0), -1)[..., 0], 0).tolist()
        rec["gd_loss_min_median"] = np.nanmedian(np.nanmin(L_, -1), 0).tolist()
        rec["gd_T_median"] = np.nanmedian(np.nanmax(z["logged_T"], -1), 0).tolist()
    for key, v in lg.items():
        per_step[f"{p.stem}/{key}"] = v.astype(np.float32)
    out["landing"][p.stem] = rec

save_json(results_dir("analysis") / f"{args.organism}.json", out)
save_npz(results_dir("analysis") / f"{args.organism}_per_step.npz", **per_step)
ks = sorted({int(k) for v in out["joint"].values() for k in v} | {int(k) for v in out["item"].values() for k in v})
for view in [v for v in ("joint", "item") if out[v]]:
    print(f"--- {view} ladder: ARC test accuracy at l-hat, mean over draws (n draws, nf = non-finite draws counted at the locked accuracy) "
          "[OBQA] tr = selection score on the draw's own k items")
    for stem, lad in out[view].items():
        cells = []
        for k in ks:
            e = lad.get(str(k))
            if not e:
                cells.append(f"k={k}:   --".ljust(40)); continue                   # as wide as a cell with its nf counts
            nf, onf = e.get("n_nonfinite"), e.get("obqa_n_nonfinite")
            tr = f"{e['train_acc_mean']:.2f}" if e["train_acc_mean"] is not None else "--"
            cells.append(f"k={k}: {e['mean']:.3f} ({e['n_draws']}{f', {nf} nf' if nf is not None else ''}) [{e.get('obqa_mean', float('nan')):.2f}{f' {onf} nf' if onf is not None else ''}] tr {tr}")
        print(f"{stem:20s} " + "  ".join(cells))
print("locked:", out.get("acc_locked"), "honest:", out.get("acc_honest"), "| obqa locked:", out.get("obqa_acc_locked"), "honest:", out.get("obqa_acc_honest"))
for name, r in out["landing"].items():
    nfc = r.get("nonfinite_count")
    print(f"{name:22s} landed {np.round(r['landed_rate'], 2).tolist()}" + (f"  non-finite rows per layer {nfc}" if nfc and any(nfc) else ""))
