"""h_{H,l,j} at every layer for the pool (80) and the ARC test (100) items in the HONEST context (prompt organism: the
HONEST instruction; ft_lock: the password PW of data.py in front of the body), the honest counterpart of
10_captures.py. Read by 36_honest_reference.py (the honest graft).
Nothing here touches the method. -> results_{cfg}/captures/{organism}/honest_{pool,test}.npz"""
import time
import numpy as np, torch
from _common import *
from actflow.readout import encode, capture
from actflow.metrics import accuracy, letter_hist

ap = parser(__doc__); ap.add_argument("--batch", type=int, default=16)
args = ap.parse_args()
cfg, splits, org, lids = setup(args)
nL = cfg["model"]["n_layers"]
out_dir = results_dir("captures", args.organism)
for which in ("pool", "test"):
    its = items(splits, which); gold = gold_of(its)
    prompts = [org.honest_prompt(q, ch) for q, ch, _ in its]
    H, LG = [], []
    t0 = time.time()
    for s in range(0, len(prompts), args.batch):
        ids, mask, pos, _ = encode(org.tok, prompts[s:s + args.batch], args.device)
        store, lg = capture(org.model, ids, mask, pos, lids, list(range(nL)), last_only=True)
        H.append(torch.stack([store[l] for l in range(nL)], 1).cpu().numpy().astype(np.float32))
        LG.append(lg.cpu().numpy())
    H, LG = np.concatenate(H), np.concatenate(LG)
    ans = LG.argmax(-1)
    log(f"honest {which}: n={len(its)} acc_honest={accuracy(ans, gold):.3f} hist={letter_hist(ans)} "
        f"{time.time()-t0:.0f}s peak={cuda_peak():.1f}GB")
    save_npz(out_dir / f"honest_{which}.npz", h=H, gold=gold, logits=LG, answer=ans, prompt=np.array(prompts),
             letter_ids=np.array(lids), acc_honest=np.float32(accuracy(ans, gold)))
