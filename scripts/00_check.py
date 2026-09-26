"""First ten minutes on a GPU machine. Measures the three rates and peak memory, and at two layers checks
  (i) slice == hook tails, (ii) F(h_S) == the locked logits, (iii) padded == single-row forward,
  (iv) the one-shot step raises the correct letter's logit on most rows, (v) peak memory below --max-peak-gb,
and prints one one-shot / ActFlow / GD landing (draws of size 1) with their errors."""
import time
import numpy as np, torch
from _common import *
from actflow.readout import encode, capture, run_tail, OneTokenTail
from actflow.graft import grafted_logits, graft_hook
from actflow.landing import raise_to, pinv_step, oneshot, actflow, gd, line_points, landing_numbers, gram_stats

ap = parser(__doc__)
ap.add_argument("--item", type=int, default=0)
ap.add_argument("--layers", nargs="+", type=int, default=[10, 20])
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--mbatch", type=int, default=80)
ap.add_argument("--max-peak-gb", type=float, default=40.0, help="exit 4 above this (check v; an A40 shows ~45 GiB)")
ap.add_argument("--max-capture-s", type=float, default=0.35, help="exit 4 if a capture row is slower (2x the A40 basis)")
ap.add_argument("--tol-pad", type=float, default=1e-3)
ap.add_argument("--tol-eq", type=float, default=1e-3)
ap.add_argument("--tol-impl", type=float, default=1e-4)
args = ap.parse_args()
cfg, splits, org, lids = setup(args)
nL, dev = cfg["model"]["n_layers"], args.device
# gate (iv): the one-shot step raises the gold logit on more than this share of rows; an organism may lower it in its config
MIN_UP = float(cfg["organisms"][args.organism].get("check_min_oneshot", 0.9))
layers = [l for l in args.layers if l < nL]
m = float(splits["margin"])
pool = items(splits, "pool"); prompts = prompts_of(org, pool); gold = gold_of(pool)
model, tok = org.model, org.tok


def sync():
    if dev == "cuda": torch.cuda.synchronize()


# 1. capture rate at batch 16 and padded == single row
B = min(args.batch, len(prompts))
ids, mask, pos, lengths = encode(tok, prompts[:B], dev)
sync(); t = time.time()
store, lg = capture(model, ids, mask, pos, lids, list(range(nL)), last_only=True)
sync(); t_cap = time.time() - t
i1, m1, p1, _ = encode(tok, [prompts[args.item]], dev)
_, lg1 = capture(model, i1, m1, p1, lids, [0])
d_pad = float((lg[args.item] - lg1[0]).abs().max())
log(f"capture: batch {B} x {ids.shape[1]} tokens  {t_cap:.2f}s = {t_cap/B:.3f}s/row  |padded - single|={d_pad:.1e}  peak={cuda_peak():.1f}GB")
assert d_pad < args.tol_pad, f"padded != single-row forward: {d_pad:.2e} (check iii)"
log(f"locked letter logits item {args.item}: {lg[args.item].tolist()} gold={gold[args.item]} acc@{B}={float((lg.argmax(-1).cpu().numpy()==gold[:B]).mean()):.2f}")

# 2. one-token tail: (F,J) rate at batch mbatch; slice == hook == locked logits
MB = min(args.mbatch, len(prompts))
sync(); t = time.time()
tail = OneTokenTail(model, tok, prompts[:MB], lids, dev, impl=args.impl)
sync(); t_pre = time.time() - t
log(f"prefill {MB} rows: {t_pre:.1f}s  cache length {tail.past.get_seq_length()} = T-1 = {tail.T-1}  peak={cuda_peak():.1f}GB")
d_pre = float((tail.logits_S[:B] - lg).abs().max())
assert d_pre < args.tol_eq, f"prefill logits != capture logits: {d_pre:.2e}"
for l in layers:
    h0 = tail.h_S[:, l]
    sync(); t = time.time(); F0, J0 = tail.FJ(h0, l); sync(); t_fj = time.time() - t
    tail.impl = "hook"; Fh = tail.F(h0, l); tail.impl = args.impl
    d_eq = float((F0 - tail.logits_S).abs().max()); d_impl = float((Fh - F0).abs().max())
    rank, sr = gram_stats(J0)
    log(f"l={l}: (F,J) at batch {MB}: {t_fj:.2f}s  |F(h_S) - locked|={d_eq:.1e}  |slice - hook|={d_impl:.1e}  "
        f"rank min {int(rank.min())}  sigma_min/max min {float(sr.min()):.1e} median {float(sr.median()):.1e}  peak={cuda_peak():.1f}GB")
    assert d_eq < args.tol_eq, f"F(h_S) != locked logits at l={l}: {d_eq:.2e} (check ii)"
    assert d_impl < args.tol_impl, f"slice != hook at l={l}: {d_impl:.2e} (check i)"
# 3. one landing of each rule at the two layers (pool items 0..MB-1; draws of size 1 = the per-item rule), with errors
gold_t = torch.as_tensor(gold[:MB], device=dev)
groups = [torch.tensor([b], device=dev) for b in range(MB)]
for l in layers:
    h0 = tail.h_S[:, l]; F0, J0 = tail.FJ(h0, l); gH = raise_to(F0, gold_t, m)
    res = {}
    sync(); t = time.time(); h = oneshot(tail, h0, l, gH, groups, F0=F0, J0=J0); F = tail.F(h, l); res["oneshot"] = (h, F, time.time() - t)
    sync(); t = time.time(); h = actflow(tail, h0, l, line_points(F0, gH, 12), groups, logged={}, gold=gold_t); F = tail.F(h, l); res["actflow12"] = (h, F, time.time() - t)
    sync(); t = time.time(); h, steps, F, _ = gd(tail, h0, l, gH, groups, max_steps=100); res["gd100"] = (h, F, time.time() - t)
    for nm, (h, F, sec) in res.items():
        err, mg, landed, _ = landing_numbers(F, gH, gold_t)
        step = (h - h0).norm(dim=-1) / h0.norm(dim=-1)
        log(f"l={l} {nm:8s}: landed {float(landed.float().mean()):.2f}  err median {float(err.median()):.2f}  "
            f"margin median {float(mg.median()):.2f}  |step|/|h0| median {float(step.median()):.3f}  {sec:.1f}s")
    log(f"l={l} actflow {res['actflow12'][2] / 12:.2f}s per step at batch {MB}, with the per-step record as in 30_manufacture")
    if "gd100" in res:
        log(f"l={l} gd steps median {float(steps.float().median()):.0f}")
    F1 = res["oneshot"][1]
    up = float((F1[torch.arange(MB), gold_t] - F0[torch.arange(MB), gold_t] > 0).float().mean())
    log(f"l={l} one shot raised the gold logit on {up:.2f} of rows (gate > {MIN_UP})")
    if not args.tiny:
        assert up > MIN_UP, f"one shot does not raise the gold logit at l={l}: {up:.3f} <= {MIN_UP} (check iv)"
del tail

# 4. graft tail rate at batch 16 (full sequence) and hook == batched rows
ids, mask, pos, _ = encode(tok, prompts[:B], dev)
store, lg = capture(model, ids, mask, pos, lids, layers, last_only=False)
l = layers[0]
u = torch.randn(2, cfg["model"]["d"], device=dev); u = u / u.norm(dim=-1, keepdim=True); tt = torch.tensor([3.0, -2.0], device=dev)
sync(); t = time.time()
lg_rows = grafted_logits(model, store[l], l, mask, pos, lids, u, tt, batch=args.batch)
sync(); t_tail = time.time() - t
with graft_hook(model, l, u[0], 3.0):
    _, lg_h = capture(model, ids, mask, pos, lids, [0])
d_g = float((lg_h - lg_rows[:, 0]).abs().max())
log(f"graft tail from l={l}: {2*B} rows in {t_tail:.2f}s = {t_tail/(2*B):.3f}s/row  |hook - rows|={d_g:.1e}  peak={cuda_peak():.1f}GB")
assert d_g < 1e-3
if cuda_peak() > args.max_peak_gb:
    log(f"STOP: peak memory {cuda_peak():.1f} GB > {args.max_peak_gb} GB at --batch {args.batch} --mbatch {args.mbatch}; lower --batch / --mbatch "
        f"(and the matching --batch of 30_manufacture / 35_graft in run_A.sh) and rerun this check"); raise SystemExit(4)
if t_cap / B > args.max_capture_s and not args.tiny:
    log(f"STOP: capture {t_cap/B:.3f} s/row > {args.max_capture_s} (2x the A40 basis); revise the cost model before continuing"); raise SystemExit(4)
# written only after every gate passed: run_A.sh skips this check when the file exists
save_json(results_dir("logs") / f"check_{args.organism}.json", {"meta": meta(args), "min_oneshot": MIN_UP, "capture_s_per_row": t_cap / B,
          "tail_s_per_row": t_tail / (2 * B), "peak_gb": cuda_peak()})
log(f"ALL CHECKS PASSED  capture {t_cap/B:.3f}s/row  (F,J)@{MB} see above  tail {t_tail/(2*B):.3f}s/row  peak {cuda_peak():.1f}GB")
