"""The honest endpoint of the organism on the two test sets (the top line of every graft plot): the same items in the
honest context (prompt organism: the HONEST instruction; ft_lock: the password PW of data.py in front of the
body). Nothing here touches the method. ft_lock: exit 3, no honest.json, when honest - locked accuracy on the ARC test
set is below --min-gap (a wrong password or a LoRA merge that did nothing). -> results_{cfg}/captures/{organism}/honest.json"""
import time
import numpy as np, torch
from _common import *
from actflow.readout import encode, capture
from actflow.metrics import accuracy, letter_hist

ap = parser(__doc__); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--min-gap", type=float, default=0.3, help="ft_lock: least honest - locked accuracy on the ARC test set "
                "(--tiny only logs it: a random model has no gap)")
args = ap.parse_args()
cfg, splits, org, lids = setup(args)
out = {"organism": args.organism, "meta": meta(args)}
for which in ("test", "test_obqa"):
    its = items(splits, which); gold = gold_of(its)
    prompts = [org.honest_prompt(q, ch) for q, ch, _ in its]
    LG = []
    t0 = time.time()
    for s in range(0, len(prompts), args.batch):
        ids, mask, pos, _ = encode(org.tok, prompts[s:s + args.batch], args.device)
        _, lg = capture(org.model, ids, mask, pos, lids, [0], last_only=True)
        LG.append(lg.cpu().numpy())
    LG = np.concatenate(LG); ans = LG.argmax(-1)
    cap = np.load(results_dir("captures", args.organism) / f"{which}.npz")
    out[which] = {"acc_honest": accuracy(ans, gold), "hist_honest": letter_hist(ans), "acc_locked": float(cap["acc_locked"]),
                  "hist_locked": letter_hist(cap["answer"]), "n": len(its)}
    log(f"{which}: honest acc {out[which]['acc_honest']:.3f} hist {out[which]['hist_honest']}  locked acc {out[which]['acc_locked']:.3f}  {time.time()-t0:.0f}s")
gap = round(out["test"]["acc_honest"] - out["test"]["acc_locked"], 6)          # acc_locked is float32: 0.6 - 0.3 must not read as 0.2999999
msg = f"honest - locked accuracy on test = {gap:.3f} < --min-gap {args.min_gap}"
if cfg["organisms"][args.organism]["kind"] == "ft_lock" and gap < args.min_gap:
    if args.tiny:
        log(f"tiny, logged only: {msg} (a random model has no gap)")
    else:
        log(f"STOP: {msg}: a wrong password or a LoRA merge that did nothing; honest.json not written"); raise SystemExit(3)
save_json(results_dir("captures", args.organism) / "honest.json", out)
