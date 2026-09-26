"""Stand-in landings for the honest graft (the "honest graft" column of the paper's Table 2). They go through 35_graft.py unchanged: its reference(h_star, h_pool_mean, draw) then returns the
honest-activation reference, and its layer selection, test and curves are the ones every ActFlow rule gets.

For every draw size k this writes one file to landing/{organism}/:
  honestk_N0_k{k}.npz   h_star[i] = h_{H,l,i}, the honest capture of pool item i at every layer l, so for draw K
                        u_l = unit(mean_K h_{H,l} - mean_U h_{S,l}),  t_l = <mean_K h_{H,l}, u_l>.
                        ActFlow's reference with the manufactured states h* of the same k items replaced by their
                        honest activations. Only the source of the k reference states changes.
No label enters u or t. The draw's k labels only pick l-hat, as for ActFlow (35_graft.py lhat_of).
honest-graft is a comparison that needs the honest context (honest instruction or password), so it runs only on locked
models whose honest activations we have. run_honest_ref.sh copies its records to results*/honest-graft/, next to graft/.
Inputs: captures/{organism}/honest_pool.npz (16_honest_captures.py) and pool.npz (10_captures.py), same last-token
block outputs. Run it under its own ACTFLOW_RESULTS root (run_honest_ref.sh does), so that these files and the grafts
never mix with the ActFlow landings that 40_analyse.py globs."""
import argparse, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from actflow.io import use_cfg, load_cfg, results_dir, save_npz, log   # noqa: E402
from actflow.reference import all_draws                               # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--organism", required=True)
ap.add_argument("--cfg", default="qwen7b")
ap.add_argument("--tiny", action="store_true", help="the tiny splits and results_tiny_{cfg}/ (CPU dry run)")
ap.add_argument("--k", nargs="+", type=int, default=None, help="draw sizes (default: every budget above 1)")
args = ap.parse_args()
import os                                                                  # noqa: E402
if not args.tiny and not os.environ.get("ACTFLOW_RESULTS", "").startswith("honestref"):
    sys.exit("run this through run_honest_ref.sh: ACTFLOW_RESULTS must be an honestref_* root, never results*/ "
             "(40_analyse.py globs the ActFlow landing folders)")
use_cfg(args.cfg, args.tiny)
splits = load_cfg("splits")
if args.tiny:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _common import TINY_SPLITS                                       # noqa: E402
    splits.update(TINY_SPLITS)
ks = args.k or [int(b) for b in splits["budgets"] if int(b) > 1]

cdir = results_dir("captures", args.organism)
hon, cap = np.load(cdir / "honest_pool.npz"), np.load(cdir / "pool.npz")
H, HS = hon["h"].astype(np.float32), cap["h"]                            # [n_pool, nL, d] each, last prompt token
assert H.shape == HS.shape, f"honest {H.shape} vs locked {HS.shape}"
assert (hon["gold"] == cap["gold"]).all(), "honest and locked pool captures list the items in different orders"
assert np.isfinite(H).all(), "non-finite honest capture"
n_pool = H.shape[0]
all_d = all_draws(splits, n_pool)
fin = np.ones(H.shape[:2], bool)
for k in ks:
    nd = sum(1 for kk, _ in all_d if kk == k)
    assert nd > 0, f"k={k} is not a budget of the splits ({splits['budgets']})"
    draw_items = np.stack([all_d[(k, dn)] for dn in range(nd)]).astype(np.int64)
    name = f"honestk_N0_k{k}"
    save_npz(results_dir("landing", args.organism) / f"{name}.npz", h_star=H, finite=fin, stem=np.array("honestk_N0"),
             rule=np.array("honestk"), N=np.int64(0), scheme=np.array("corrected"), path=np.array("line"), rank_spec=np.array("honest"),
             k=np.int64(k), draw_items=draw_items, gold=cap["gold"],
             provenance=np.array(f"36_honest_reference.py: h_star = honest captures of the draw's own items (captures/{args.organism}/honest_pool.npz)"))
    log(f"{args.cfg} {args.organism}: wrote {name} ({nd} draws of {k})")
