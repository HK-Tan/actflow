"""Every data figure and table of the paper, read from the result files (laptop, no GPU).
Reads
  results*/graft/{organism}/{stem}_k{k}/k{k'}_d{d}.json       ActFlow and GD grafts (35_graft.py)
  results*/honest-graft/{organism}/honestk_N0_k{k}/...         the honest graft (run_honest_ref.sh)
  results*/sft/{organism}/sft_k{k}.json                        fine-tuning (70_sft.py), at the last (1,024-example) checkpoint
  results*/captures/{organism}/honest.json, pool.npz           locked and honest accuracies (15_honest.py), pool activations
  results*/landing/{organism}/*.npz                            landing errors, ranks and target logits (30_manufacture.py)
  results*/logs/, results_qwen7b/logs/pods/                    job logs and GPU types (the GPU-hours)
Writes to --out (default results_paper/)
  tab_main.tex      Table 2: held-out accuracy of every method on the six locked models at k = 4, 10, 40
  fig_ladder.pdf    Figure 3: (a) accuracy against k, (b) per-item against shared shifts, (c) steps, schemes, paths
  tab_general.tex   Appendix B: the configs of Figure 3c at k = 4, 10, 40
  tab_locks.tex     Appendix B: locked and honest accuracy of the ten locked models
  tab_seeds.tex     Appendix B: the Qwen LoRA locks of seeds 1 to 5
  tab_compute.tex   Appendix B: GPU-hours of every step
  tab_full_*.tex    Appendix B: accuracy and landing error on the labeled items, chosen layers, balanced accuracy and
                    letter shares, for every rule on the ten locked models
  tab_af01.tex, tab_lastlayer.tex, tab_window.tex   Appendix B: AF_0.1, the last layer, the window of the locks' source paper
and prints the numbers quoted in the text (81_numbers.py prints the others).
The paper folder runs this with --out iclr-submission/figures (its build_results.py)."""
import argparse, glob, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default=None, help="output folder (default: results_paper/)")
args = ap.parse_args()
A = Path(__file__).resolve().parents[1]                                # the actflow repo
OUT = Path(args.out).resolve() if args.out else A / "results_paper"
OUT.mkdir(parents=True, exist_ok=True)
ROOT = {"qwen": A / "results_qwen7b", "llama": A / "results_llama8b", "mistral": A / "results_mistral7b"}
SFT = ROOT
HONEST = {m: r / "honest-graft" for m, r in ROOT.items()}              # graft with honest activations (needs the honest context)
CELLS = [("qwen", "prompt", "Qwen2.5-7B", "prompt"), ("qwen", "ft_lock", "Qwen2.5-7B", "LoRA"),
         ("llama", "prompt", "Llama-3-8B", "prompt"), ("llama", "ft_lock", "Llama-3-8B", "LoRA"),
         ("mistral", "prompt", "Mistral-7B", "prompt"), ("mistral", "ft_lock", "Mistral-7B", "LoRA")]
KS = [4, 10, 40]


def recs(m, org, stem, k, kp=None):
    kp = kp or k
    gdir = HONEST[m] if stem.startswith("honest") else ROOT[m] / "graft"
    return [json.load(open(f)) for f in sorted(glob.glob(str(gdir / org / f"{stem}_k{k}" / f"k{kp}_d*.json")))]


def acc(m, org, stem, k, kp=None, key="test_acc"):
    r = recs(m, org, stem, k, kp)
    v = [x[key] for x in r if x.get(key) is not None]
    assert r and len(v) == len(r), (m, org, stem, k, kp, len(r), len(v))
    return float(np.mean(v))


def sft(m, org, k, key="test"):
    d = json.load(open(SFT[m] / "sft" / org / f"sft_k{k}.json"))
    return float(d["mean_by_checkpoint"]["1024"][key])


def correct(m, org, stem, k, obqa=False, kp=None):
    """Draws x test items matrix of 0/1 correctness at the chosen layer (FT: the 1,024-example checkpoint)."""
    if stem == "ft":
        d = json.load(open(SFT[m] / "sft" / org / f"sft_k{k}.json"))
        C = np.array([x["ckpt"]["1024"]["test_obqa" if obqa else "test"]["correct"] for x in d["draws"]], float)
        assert abs(C.mean() - sft(m, org, k, "test_obqa" if obqa else "test")) < 1e-6
    else:
        C = np.array([x[("obqa_" if obqa else "") + "test_correct"] for x in recs(m, org, stem, k, kp)], float)
        assert abs(C.mean() - acc(m, org, stem, k, kp, key=("obqa_" if obqa else "") + "test_acc")) < 1e-6
    return C


def draw_se(C):
    """Standard error of the mean accuracy over the draws (each draw's accuracy on the 100 test items is one run)."""
    a = C.mean(1)
    return float(a.std(ddof=1) / np.sqrt(len(a)))


def ref(m, org, which):
    r = recs(m, org, "af5_N40", 40)[0]
    return r["acc_locked"] if which == "locked" else r["acc_honest"]


from decimal import Decimal, ROUND_HALF_UP


def f3(x):
    """Two decimals, half up (k=40 means over 2 draws often end in 5), no leading zero."""
    r = Decimal(str(round(float(x), 6))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return "1.00" if r >= 1 else f"{r:.2f}"[1:]


def pm(C, mark=None):
    """Mean with the standard error over draws as a subscript; mark 'bf' bolds and 'ul' underlines the mean."""
    m = {"bf": f"\\mathbf{{{f3(C.mean())}}}", "ul": f"\\underline{{{f3(C.mean())}}}"}.get(mark, f3(C.mean()))
    return f"${m}_{{\\pm {f3(draw_se(C))}}}$"


def pm_row(Cs, compete):
    """pm for each column. Among the `compete` columns the highest displayed (two-decimal) mean is bold and the second
    highest distinct value is underlined, ties included."""
    shown = {s_: float(f3(Cs[s_].mean())) for s_ in compete}
    vals = sorted(set(shown.values()), reverse=True)
    mark = {vals[0]: "bf", **({vals[1]: "ul"} if len(vals) > 1 else {})}
    return {s_: pm(C, mark.get(shown[s_]) if s_ in compete else None) for s_, C in Cs.items()}


# ---------------- Table 2: k = 4, 10, 40 -------------------------------------------------------------------------------
rows, prev = [], None
print("== Table 2 (ARC and OBQA, k = 4, 10, 40)")
for g, (m, org, model, lock) in enumerate(CELLS):
    for k in KS:
        METHODS = ("ft", "af5_N40", "afpr_N40", "af_N40", "gd_N1000")        # bold = best of these (not the honest graft)
        c = pm_row({s_: correct(m, org, s_, k) for s_ in ("honestk_N0",) + METHODS}, METHODS)
        c_o = pm_row({s_: correct(m, org, s_, k, obqa=True) for s_ in METHODS}, METHODS)
        print(model, lock, k, c, c_o)
        if k == KS[0]:
            name = f"{model} & {lock}" if model != prev else f" & {lock}"
            fixed = f"{f3(ref(m, org, 'locked'))} & {f3(ref(m, org, 'honest'))}"
            h_o = f3(recs(m, org, "af5_N40", 40)[0]["obqa_acc_honest"])
        else:
            name, fixed, h_o = " & ", " & ", ""
        prev = model
        rows.append(f"{name} & {k} & {fixed} & {c['honestk_N0']} & {c['ft']} & {c['af5_N40']} & {c['afpr_N40']} & "
                    f"{c['af_N40']} & {c['gd_N1000']} & {h_o} & {c_o['ft']} & {c_o['af5_N40']} & {c_o['afpr_N40']} & {c_o['af_N40']} & "
                    f"{c_o['gd_N1000']} \\\\")
    if g < len(CELLS) - 1:
        rows.append(r"\midrule" if CELLS[g + 1][0] != m else r"\addlinespace[2pt]")
tab = r"""\begin{table}[t]
\centering
\caption{Held-out accuracy with $k$ labeled items on 100 ARC-Easy and 100 OpenBookQA test items. Each entry is the mean over disjoint draws of the 80-item pool at $k=4$, $10$ and $40$, and subscripts are standard errors over the draws. Honest graft applies the graft \eqref{eq:graft} with the honest residual streams of the same $k$ items in place of the manufactured ones. FT is LoRA fine-tuning on the same $k$ items, and AF is ActFlow. In each row and test set, excluding honest graft, bold marks the highest accuracy and underline the second highest.}
\label{tab:main}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llcccc|ccccc|cccccc}
\toprule
 & & & \multicolumn{8}{c|}{ARC-Easy} & \multicolumn{6}{c}{OpenBookQA} \\
model & lock & $k$ & locked & honest & honest graft & FT & AF$_5$ & AF$_\PR$ & AF$_{\mathrm{full}}$ & GD & honest & FT & AF$_5$ & AF$_\PR$ & AF$_{\mathrm{full}}$ & GD \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
(OUT / "tab_main.tex").write_text(tab)

# ---------------- k = 4, 10 numbers for the text ------------------------------------------------------------------------
print("== ladder numbers (ARC)")
for m, org, model, lock in CELLS:
    print(model, lock, "FT", [round(sft(m, org, k), 3) for k in KS], "AF5", [round(acc(m, org, "af5_N40", k), 3) for k in KS],
          "AFPR", [round(acc(m, org, "afpr_N40", k), 3) for k in KS], "AFfull", [round(acc(m, org, "af_N40", k), 3) for k in KS],
          "GD", [round(acc(m, org, "gd_N1000", k), 3) for k in KS], "FT train", round(sft(m, org, 4, "train_acc"), 3))

# ---------------- Figure 3: accuracy against k -----------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig = plt.figure(figsize=(9.4, 3.7))
gs = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, 0.22, 1.75, 0.1, 1.45], wspace=0.14, hspace=0.3)
axs = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])
LINES = [("FT", None, "k", "o"), (r"AF$_5$", "af5_N40", "tab:red", "s"), (r"AF$_{\rm PR}$", "afpr_N40", "tab:green", "^"),
         (r"AF$_{\rm full}$", "af_N40", "tab:orange", "v"), ("GD", "gd_N1000", "tab:blue", "D")]
def ref_lines(ax, m, org, label=None):
    """Honest and locked accuracy (dotted) and chance (dashed); label='in' writes the names above the lines at the right
    end of the panel, label='out' writes them just right of the panel."""
    for name, val, ls in (("honest", ref(m, org, "honest"), ":"), ("chance", 0.25, "--"), ("locked", ref(m, org, "locked"), ":")):
        ax.axhline(val, color="0.5", ls=ls, lw=0.7)
        if label == "in":
            ax.text(0.99, val + 0.012, name, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=5.5, color="0.35")
        elif label == "out":
            ax.text(1.03, val, name, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=5.5, color="0.35")


for j, (m, org, model, lock) in enumerate(CELLS):
    ax = axs[j % 2, j // 2]
    for lab, stem, col, mk in LINES:
        Cs = [correct(m, org, stem or "ft", k) for k in KS]
        y, e = [C.mean() for C in Cs], [draw_se(C) for C in Cs]
        ax.errorbar(KS, y, yerr=e, color=col, marker=mk, ms=3, lw=1, capsize=1.5, label=lab)
    ref_lines(ax, m, org)
    ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS]); ax.minorticks_off()
    ax.set_xlim(KS[0] / 1.12, KS[-1] * 1.12)   # error bars drawn on an already-log axis would otherwise pull the limit toward 0
    ax.set_ylim(-0.02, 1.08)                   # columns = models, rows = locks
    if j % 2 == 1: ax.set_xlabel("labeled items $k$")
    else: ax.set_xticklabels([]); ax.set_title(model, fontsize=7.5)
    if j // 2 == 0: ax.set_ylabel(f"{lock} lock\nARC test accuracy")
    else: ax.set_yticklabels([])
axs[1, 1].legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, -0.33), ncol=5, fontsize=6.5,
                 frameon=False, columnspacing=1.2, handlelength=1.8)
axs[0, 0].text(-0.42, 1.13, "(a)", transform=axs[0, 0].transAxes, fontweight="bold", fontsize=8)

# (b): one item, and at k = 4, 40 one shift per item (per-item) vs one shift shared by the k items (shared) (Qwen, ARC)
print("== stacking (Qwen, ARC)")
gold = {org: np.load(ROOT["qwen"] / "captures" / org / "pool.npz")["gold"] for org in ("prompt", "ft_lock")}
BARS = [("GD", "gd_N1000", "gd_N1000", "tab:blue"), (r"AF$_{\rm full}$", "af_N40", "af_N40", "tab:orange"),
        (r"AF$_5$", "af_N40", "af5_N40", "tab:red")]            # one-item AF5 = AF_full (the Jacobian has 4 rows)
XB = np.array([0, 1.4, 2.8, 4.35, 5.75])                          # one item | k=4 per-item, shared | k=40 per-item, shared
for r, (org, lock) in enumerate((("prompt", "prompt"), ("ft_lock", "LoRA"))):
    ax = fig.add_subplot(gs[r, 4])
    for b, (lab, stem1, stemK, col) in enumerate(BARS):
        Cs = [correct("qwen", org, stem1, 1)] + [C for k in (4, 40) for C in (correct("qwen", org, stem1, 1, kp=k), correct("qwen", org, stemK, k))]
        r1 = recs("qwen", org, stem1, 1)
        same = sum(int(np.bincount(np.array(x["test_answers"]), minlength=4).argmax() == gold[org][x["items"][0]]) for x in r1)
        print(org, lab, "k=1", round(Cs[0].mean(), 3), "per-item4", round(Cs[1].mean(), 3), "shared4", round(Cs[2].mean(), 3),
              "per-item40", round(Cs[3].mean(), 3), "shared40", round(Cs[4].mean(), 3), "top letter = example's", f"{same}/{len(r1)}")
        ax.bar(XB + (b - 1) * 0.28, [C.mean() for C in Cs], 0.28, yerr=[draw_se(C) for C in Cs], color=col,
               error_kw=dict(lw=0.8, capsize=1.5), label=lab)
    ref_lines(ax, "qwen", org)
    ax.set_xticks(XB); ax.set_ylim(-0.02, 1.08); ax.set_xlim(XB[0] - 0.55, XB[-1] + 0.55); ax.set_yticklabels([])
    ax.set_xticklabels(["", "per-item", "shared", "per-item", "shared"] if r == 1 else [], fontsize=6)
    if r == 0:
        ax.set_title("Qwen2.5-7B", fontsize=7.5)
        ax.text(-0.08, 1.13, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=8)
    else:
        for x, t in ((XB[0], "$k=1$"), (XB[1:3].mean(), "$k=4$"), (XB[3:].mean(), "$k=40$")):
            ax.text(x, -0.2, t, transform=ax.get_xaxis_transform(), ha="center", va="top")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.33), ncol=3, fontsize=6.5, frameon=False, columnspacing=1.2, handlelength=1.2)
# (c): scheme, step count and path, Qwen, k = 10 (was Table 3)
print("== general properties (Qwen, ARC, k=10)")
G = [("GN\n$N$=1", "N1"), ("GN\n$N$=8", "N8"), ("GN\n$N$=40", "N40"), ("Euler\n$N$=40", "N40_euler"), ("seq.\n$N$=40", "N40_seq")]
for r, (org, lock) in enumerate((("prompt", "prompt"), ("ft_lock", "LoRA"))):
    ax = fig.add_subplot(gs[r, 6])
    for b, (lab, rule, col) in enumerate(((r"AF$_5$", "af5", "tab:red"), (r"AF$_{\rm PR}$", "afpr", "tab:green"))):   # the partial lifts
        Cs = [correct("qwen", org, f"{rule}_{sfx}", 10) for _, sfx in G]
        print(org, rule, [f"{C.mean():.3f}+-{draw_se(C):.3f}" for C in Cs])
        ax.bar(np.arange(len(G)) + (b - 0.5) * 0.36, [C.mean() for C in Cs], 0.36, yerr=[draw_se(C) for C in Cs], color=col,
               error_kw=dict(lw=0.8, capsize=1.5), label=lab)
    ref_lines(ax, "qwen", org, "out")            # same Qwen lines as (b) in this row
    ax.set_xticks(range(len(G))); ax.set_ylim(-0.02, 1.08); ax.set_xlim(-0.6, len(G) - 0.4); ax.set_yticklabels([])
    ax.set_xticklabels([g for g, _ in G] if r == 1 else [], fontsize=6)
    if r == 0:
        ax.set_title("Qwen2.5-7B, $k=10$", fontsize=7.5)
        ax.text(-0.08, 1.13, "(c)", transform=ax.transAxes, fontweight="bold", fontsize=8)
    else:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.33), ncol=2, fontsize=6.5, frameon=False, columnspacing=1.2, handlelength=1.2)
# App. B table: the configs of (c) at k = 4, 10, 40 (Qwen, ARC)
print("== general properties (Qwen, ARC, k=4,10,40)")
grows = []
for org, lock in (("prompt", "prompt"), ("ft_lock", "LoRA")):
    for rule, rlab in (("af5", r"AF$_5$"), ("afpr", r"AF$_\PR$")):
        for k in KS:
            Cs = [correct("qwen", org, f"{rule}_{sfx}", k) for _, sfx in G]
            print(org, rule, k, [f3(C.mean()) for C in Cs])
            head = (f"{lock} & {rlab}" if rule == "af5" else f" & {rlab}") if k == KS[0] else " & "
            grows.append(f"{head} & {k} & " + " & ".join(pm(C) for C in Cs) + r" \\")
        if rule == "af5": grows.append(r"\addlinespace[2pt]")
    if org == "prompt": grows.append(r"\midrule")
tab = r"""\begin{table}[h]
\centering
\caption{ARC-Easy test accuracy of \AF{5} and \AF{\PR} on the two Qwen locks for the configs of Figure~\ref{fig:ladder}c at $k=4$, $10$ and $40$. GN is the Gauss--Newton correction and seq.\ the sequential path. Subscripts are standard errors over draws.}
\label{tab:general}
\begin{tabular}{llcccccc}
\toprule
lock & rule & $k$ & GN $N{=}1$ & GN $N{=}8$ & GN $N{=}40$ & Euler $N{=}40$ & seq.\ $N{=}40$ \\
\midrule
""" + "\n".join(grows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(OUT / "tab_general.tex").write_text(tab)
fig.savefig(OUT / "fig_ladder.pdf", bbox_inches="tight"); fig.savefig(OUT / "fig_ladder.png", dpi=200, bbox_inches="tight")

# ---------------- App. B: locked and honest accuracy of the ten locked models ---------------------------------------
LOCKS = [("qwen", "prompt", "Qwen2.5-7B", "prompt", ""), ("qwen", "ft_lock", "", "LoRA", "1"),
         ("qwen", "ft_lock_s2", "", "LoRA", "2"), ("qwen", "ft_lock_s3", "", "LoRA", "3"), ("qwen", "ft_lock_s4", "", "LoRA", "4"),
         ("qwen", "ft_lock_s5", "", "LoRA", "5"), ("llama", "prompt", "Llama-3-8B", "prompt", ""), ("llama", "ft_lock", "", "LoRA", "1"),
         ("mistral", "prompt", "Mistral-7B", "prompt", ""), ("mistral", "ft_lock", "", "LoRA", "1")]
print("== locked and honest accuracy (ARC, OBQA)")
lrows = []
for j, (m, org, model, lock, seed) in enumerate(LOCKS):
    h = json.load(open(ROOT[m] / "captures" / org / "honest.json"))
    v = [h["test"]["acc_locked"], h["test"]["acc_honest"], h["test_obqa"]["acc_locked"], h["test_obqa"]["acc_honest"]]
    print(model or "  ", lock, seed, [round(x, 3) for x in v])
    if j and model:
        lrows.append(r"\midrule")
    lrows.append(f"{model} & {lock} & {seed} & " + " & ".join(f3(x) for x in v) + r" \\")
tab = r"""\begin{table}[h]
\centering
\caption{Accuracy of the ten locked models on the 100 ARC-Easy and 100 OpenBookQA test items. Locked is the model with the locked prompt, and honest is the same model with the honest instruction (prompt lock) or the password (LoRA lock). Seed 1 is the LoRA lock of Table~\ref{tab:main} and Figure~\ref{fig:ladder}.}
\label{tab:locks}
\begin{tabular}{lllcccc}
\toprule
 & & & \multicolumn{2}{c}{ARC-Easy} & \multicolumn{2}{c}{OpenBookQA} \\
model & lock & seed & locked & honest & locked & honest \\
\midrule
""" + "\n".join(lrows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(OUT / "tab_locks.tex").write_text(tab)

# ---------------- App. B: the Qwen LoRA locks of seeds 1 to 5 (seed 1 = the Qwen LoRA lock of Table 2) ---------------
print("== Qwen LoRA seeds (ARC; OBQA)")
SEEDS = [("ft_lock", "1"), ("ft_lock_s2", "2"), ("ft_lock_s3", "3"), ("ft_lock_s4", "4"), ("ft_lock_s5", "5")]
AFS = ("af5_N40", "afpr_N40", "af_N40", "gd_N1000")
srows = []
for j, (org, seed) in enumerate(SEEDS):
    for k in KS:
        c = pm_row({s_: correct("qwen", org, s_, k) for s_ in ("honestk_N0",) + AFS}, AFS)
        c_o = pm_row({s_: correct("qwen", org, s_, k, obqa=True) for s_ in AFS}, AFS)
        print(seed, k, {s_: round(acc("qwen", org, s_, k), 3) for s_ in ("honestk_N0",) + AFS},
              "AF5 - AFPR", round(acc("qwen", org, "af5_N40", k) - acc("qwen", org, "afpr_N40", k), 3))
        srows.append(f"{seed if k == KS[0] else ''} & {k} & {c['honestk_N0']} & " + " & ".join(c[s_] for s_ in AFS) + " & "
                     + " & ".join(c_o[s_] for s_ in AFS) + r" \\")
    if j < len(SEEDS) - 1:
        srows.append(r"\addlinespace[2pt]")
tab = r"""\begin{table}[h]
\centering
\caption{Held-out accuracy on the Qwen2.5-7B LoRA locks of seeds 1 to 5. Entries and subscripts are as in Table~\ref{tab:main}, and seed 1 is the Qwen LoRA lock of Table~\ref{tab:main}. The honest graft has ARC-Easy accuracy only. In each row and test set, bold marks the highest accuracy among \AF{5}, \AF{\PR}, \AF{\mathrm{full}} and GD and underline the second highest. Locked and honest accuracies are in Table~\ref{tab:locks}.}
\label{tab:seeds}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llc|cccc|cccc}
\toprule
 & & \multicolumn{5}{c|}{ARC-Easy} & \multicolumn{4}{c}{OpenBookQA} \\
seed & $k$ & honest graft & AF$_5$ & AF$_\PR$ & AF$_{\mathrm{full}}$ & GD & AF$_5$ & AF$_\PR$ & AF$_{\mathrm{full}}$ & GD \\
\midrule
""" + "\n".join(srows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
(OUT / "tab_seeds.tex").write_text(tab)

# ---------------- Section 5.1: time to fit the shift, GD against one ActFlow rule (Llama and Mistral locks) -----------
# 30_manufacture.py prints one line per (batch, layer) ending in its wall-clock seconds; the AF slice runs AF_full, AF_PR
# and AF_5 together, the GD slice runs GD alone, both on the same machine. Model loading and grafting are not counted.
import re
LINE = re.compile(r"^\S+\s+k=\s*\d+ l=\s*\d+ .* ([0-9.]+)s$")
def landing_minutes(log):
    return sum(float(m_.group(1)) for m_ in map(LINE.match, open(log).read().splitlines()) if m_) / 60
print("== landing time (min): AF slice (3 rules), GD slice, GD / (AF per rule)")
tot_af = tot_gd = 0.0; ratios = []
for m in ("llama", "mistral"):
    for org in ("prompt", "ft_lock"):
        for k in KS:
            af = landing_minutes(ROOT[m] / "logs" / f"{org}_B_rules_af_afpr_af5_k_{k}.log")
            gd_ = landing_minutes(ROOT[m] / "logs" / f"{org}_B_rules_gd_k_{k}.log")
            tot_af += af; tot_gd += gd_; ratios.append(gd_ / (af / 3))
            print(m, org, k, f"{af:.1f} {gd_:.1f} {gd_ / (af / 3):.1f}x")
print(f"total: AF 3 rules {tot_af / 60:.1f} GPU-h, GD {tot_gd / 60:.1f} GPU-h, GD / AF per rule {tot_gd / (tot_af / 3):.1f}x, "
      f"per slice {min(ratios):.1f}x to {max(ratios):.1f}x")

# ---------------- App. B: GPU-hours of every step (the wall clock of each job, one job per GPU) ------------------------
import datetime as dt
from collections import defaultdict
HDR = re.compile(r"^== run_A cfg=\S+ organism=(\S+) block=\S+ args=(.*) tag=(\S+) (\S+Z)$")
DONE = re.compile(r"^== done \S+ (\S+Z)$")
LANDED = re.compile(r"^(\d\d:\d\d:\d\d) done; \d+ landings to graft")
PREP = re.compile(r"^== prep done \S+ (\S+Z)$")
HG = re.compile(r"^== (\S+Z) (?:done )?cfg=")
GMIN = re.compile(r"^\d\d:\d\d:\d\d (\S+): done ([0-9.]+) min")
SFT_EV = re.compile(r"^== GPU \d+ (start|done ) (\S+) (\S+) sft k=(\d+) (\S+Z)$")
HONEST_CAP = re.compile(r"^== (?:start \S+ \S+|honest all done) (\d\d:\d\d:\d\d)$")
iso = lambda s: dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
gpu_of = lambda f: json.load(open(f)).get("meta", {}).get("gpu") or json.load(open(f))["gpu"]


def at(anchor, hms):
    """The first time of day hms at or after anchor - 12 h (log stamps carry no date and may cross midnight)."""
    t = anchor.replace(hour=int(hms[:2]), minute=int(hms[3:5]), second=int(hms[6:8]))
    return t + dt.timedelta(days=1) if t < anchor - dt.timedelta(hours=12) else t


H = defaultdict(float)                                                  # (step, GPU) -> seconds
for m, R in ROOT.items():
    for f in sorted((R / "logs").glob("*.log")):
        L = open(f, errors="replace").read().splitlines()
        if f.name.endswith("_prep.log"):                                # 00_check, 10_captures, 15_honest (re-entries last 0 s)
            gpu = gpu_of(R / "captures" / f.stem[:-5] / "meta.json")
            for i, x in enumerate(L):
                if (h := HDR.match(x)) and (d := next((PREP.match(y) for y in L[i + 1:] if PREP.match(y) or HDR.match(y)), None)):
                    j = next(j for j in range(i + 1, len(L)) if PREP.match(L[j]))
                    dl = max([60 * int(a) + int(b) for y in L[i + 1:j] if "Fetching" in y     # a model download inside the job
                              for a, b in re.findall(r"\[(\d+):(\d+)<", y)], default=0)    # (progress lines: take the last)
                    H["prep", gpu] += (iso(d.group(1)) - iso(h.group(4))).total_seconds() - dl
        elif f.name.startswith("honest_graft_"):                        # the honest graft, 36_honest_reference.py + 35_graft.py
            t = [iso(h.group(1)) for h in map(HG.match, L) if h]
            other = sum(float(g.group(2)) * 60 for g in map(GMIN.match, L) if g and not g.group(1).startswith("honestk"))
            H["honest graft", gpu_of(R / "honest-graft" / f.stem[13:] / "meta.json")] += (t[-1] - t[0]).total_seconds() - other
        elif not f.name.startswith("sft_"):                             # run_A: landings, then their grafts, in one job
            h = next(HDR.match(x) for x in L if HDR.match(x))
            t0, t1 = iso(h.group(4)), iso(next(DONE.match(x) for x in L if DONE.match(x)).group(1))
            tl = at(t0, next(LANDED.match(x) for x in L if LANDED.match(x)).group(1))
            gd = "--rules gd" in h.group(2)
            gpu = gpu_of(R / "landing" / h.group(1) / f"meta_{h.group(3)}.json")
            H["GD landing" if gd else "AF landing", gpu] += (tl - t0).total_seconds()
            H["GD graft" if gd else "AF graft", gpu] += (t1 - tl).total_seconds()
ev = {}
for x in open(ROOT["qwen"] / "logs" / "pods" / "L40Sx4-sft" / "sft_run.out").read().splitlines():
    if e := SFT_EV.match(x):
        ev.setdefault(e.groups()[1:4], {})[e.group(1).strip()] = iso(e.group(5))
for (cfg, org, k), v in ev.items():
    m = {"qwen7b": "qwen", "llama8b": "llama", "mistral7b": "mistral"}[cfg]
    H["fine-tuning", gpu_of(ROOT[m] / "sft" / org / f"sft_k{k}.json")] += (v["done"] - v["start"]).total_seconds()
t = [dt.datetime.strptime(h.group(1), "%H:%M:%S") for h in
     map(HONEST_CAP.match, open(ROOT["qwen"] / "logs" / "pods" / "L40S-honest" / "honest.out").read().splitlines()) if h]
H["honest captures", "NVIDIA L40S"] += (t[-1] - t[0]).total_seconds()  # 16_honest_captures.py, the ten locked models in turn on one L40S
STEPS = [("prep", "Locked activations and accuracies"), ("AF landing", "ActFlow runs"), ("GD landing", "GD runs"),
         ("AF graft", "Grafts of the ActFlow shifts"), ("GD graft", "Grafts of the GD shifts"),
         ("honest captures", "Honest activations"), ("honest graft", "Honest graft"), ("fine-tuning", "Fine-tuning")]
GPUS = ["NVIDIA A40", "NVIDIA L40S"]
assert {g for _, g in H} <= set(GPUS), H.keys()
print("== GPU-hours by step (A40, L40S, total)")
crows = []
for s_, lab in STEPS + [(None, "Total")]:
    v = [sum(sec for (s2, g), sec in H.items() if g == gg and s_ in (s2, None)) / 3600 for gg in GPUS]
    print(lab, [round(x, 2) for x in v], round(sum(v), 2))
    crows.append((r"\midrule" + "\n" if s_ is None else "") + f"{lab} & " + " & ".join(f"{x:.2f}" if x else "--" for x in v + [sum(v)]) + r" \\")
tab = r"""\begin{table}[h]
\centering
\caption{GPU-hours of each stage of the experiments, as the wall-clock time of each job summed over jobs, with one job per GPU. The first stage includes checks that the code reproduces the locked logits. ActFlow runs include every rule and config of Appendix~\ref{apd:actflow-setup}, and grafts include layer selection and the evaluation on both test sets. Model downloads and idle time are not counted.}
\label{tab:compute}
\begin{tabular}{lccc}
\toprule
stage & A40 & L40S & total \\
\midrule
""" + "\n".join(crows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
(OUT / "tab_compute.tex").write_text(tab)

# ---------------- App. B: full results on ARC for every rule and all ten locked models ---------------------------------
NL = {"qwen": 28, "llama": 32, "mistral": 32}
EPS = {"qwen": 1e-6, "llama": 1e-5, "mistral": 1e-5}                # RMSNorm epsilon of the final norm
RULES = ["af5_N40", "afpr_N40", "af_N40", "gd_N1000"]
HON = "honestk_N0"
RULE_HEAD = r"AF$_5$ & AF$_\PR$ & AF$_{\mathrm{full}}$ & GD"
FULL = [(m, org, model, lock, seed, k) for m, org, model, lock, seed in LOCKS
        for k in ([1] if m == "qwen" and org in ("prompt", "ft_lock") else []) + KS]  # k = 1 only on the two Qwen locks of Table 2


def cell(m, org, stem, k):
    """Graft records of one (locked model, rule, k), or None where the rule was not run (AF_5 = AF_full at k = 1)."""
    return None if k == 1 and stem in ("af5_N40", HON) else recs(m, org, stem, k)


def land_err(m, org, stem, k):
    """Median over the labeled items of max_a |F_a - gamma_H,a| at the chosen layer of the item's draw."""
    e = np.load(ROOT[m] / "landing" / org / f"{stem}_k{k}.npz")["landing_err"]
    return float(np.median(np.concatenate([e[r["items"], r["lhat"]] for r in recs(m, org, stem, k)])))


def lhat_str(r):
    l = np.array([x["lhat"] for x in r]) + 1                            # paper layers are numbered from 1
    med = f"{np.median(l):g}"
    return med if l.min() == l.max() else f"{med} ({l.min()}--{l.max()})"


def head(i):
    """Model, lock and seed on the first row of a locked model, blank below."""
    m, org, model, lock, seed, k = FULL[i]
    first = i == 0 or FULL[i - 1][1] != org or FULL[i - 1][0] != m
    top = first and (i == 0 or FULL[i - 1][0] != m)
    return (r"\midrule" + "\n" if top and i else r"\addlinespace[2pt]" + "\n" if first and i else "") + \
           f"{model if top else ''} & {lock if first else ''} & {seed if first else ''} & {k}"


def full_tab(label, caption, cols, body, spec):
    return (r"""\begin{table}[h]
\centering
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lllc""" + spec + r"""}
\toprule
""" + cols + r"""
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")


D = "--"
GOLD_SHARE = np.bincount(np.load(ROOT["qwen"] / "captures" / "prompt" / "test.npz")["gold"], minlength=4) / 100
print("== full results: labeled-item accuracy (AF5 AFPR AFfull GD honest) | landing error at l-hat (AF5 AFPR AFfull GD)")
fit, lay, let = [], [], []
for i, (m, org, model, lock, seed, k) in enumerate(FULL):
    R_ = {s_: cell(m, org, s_, k) for s_ in RULES + [HON]}
    tr = [f3(np.mean([x["sel_acc"][x["lhat"]] for x in R_[s_]])) if R_[s_] else D for s_ in RULES + [HON]]
    er = [f"{land_err(m, org, s_, k):.2f}" if R_[s_] else D for s_ in RULES]
    print(model or "  ", lock, seed, k, tr, er)
    fit.append(f"{head(i)} & " + " & ".join(tr + er) + r" \\")
    lay.append(f"{head(i)} & " + " & ".join(lhat_str(R_[s_]) if R_[s_] else D for s_ in RULES + [HON]) + r" \\")
    let.append(f"{head(i)} & " + " & ".join([f3(np.mean([x["balanced_acc"] for x in R_[s_]])) if R_[s_] else D for s_ in RULES]
                                               + [f3(np.mean([max(x["letter_hist"]) for x in R_[s_]])) if R_[s_] else D for s_ in RULES]) + r" \\")
ID = r"model & lock & seed & $k$"
(OUT / "tab_full_fit.tex").write_text(full_tab(
    "tab:full-fit",
    r"Fit on the $k$ labeled items. Accuracy is the share of the $k$ labeled items that the graft at the chosen layer $\hat\ell$ answers correctly, averaged over draws. Error is the median over the 80 pool items of $\|F_{S,\hat\ell,i}(h_{S,\hat\ell,i}+x_N)-\gamma_{H,i}\|_\infty$, the largest distance of the four logits of item $i$ from its target after the last step at the layer $\hat\ell$ of its draw, before the graft. Dashes mark \AF{5} at $k=1$, where it equals \AF{\mathrm{full}}, and the honest graft at $k=1$, which was not run.",
    r" & & & & \multicolumn{5}{c|}{accuracy on the labeled items} & \multicolumn{4}{c}{error at $\hat\ell$} \\" + "\n" + ID + " & " + RULE_HEAD + r" & honest graft & " + RULE_HEAD + r" \\",
    fit, "|ccccc|cccc"))
(OUT / "tab_full_layers.tex").write_text(full_tab(
    "tab:full-layers",
    r"Chosen layer $\hat\ell$ as the median over draws, with the smallest and largest $\hat\ell$ in parentheses when they differ. Layers are numbered from $1$ to $n_{\rm L}$, which is $28$ for Qwen and $32$ for Llama and Mistral. Dashes are as in Table~\ref{tab:full-fit}.",
    ID + " & " + RULE_HEAD + r" & honest graft \\", lay, "|ccccc"))
(OUT / "tab_full_letters.tex").write_text(full_tab(
    "tab:full-letters",
    r"Balanced accuracy and top share on the 100 ARC-Easy test items, averaged over draws. Balanced accuracy is the mean over the four correct letters of the accuracy on the test items with that correct letter. Top share is the share of test answers that are the most frequent answer letter of the draw. The correct letters A, B, C and D have shares " + ", ".join(f"${f3(x)}$" for x in GOLD_SHARE[:3]) + f" and ${f3(GOLD_SHARE[3])}$ among the test items. Dashes are as in Table~\\ref{{tab:full-fit}}.",
    r" & & & & \multicolumn{4}{c|}{balanced accuracy} & \multicolumn{4}{c}{top share} \\" + "\n" + ID + " & " + RULE_HEAD + " & " + RULE_HEAD + r" \\",
    let, "|cccc|cccc"))

# AF_0.1, run only on the two Qwen locks of Table 2
print("== AF_0.1 (Qwen): ARC, OBQA, labeled, error, l-hat, balanced, top share")
arows = []
for org, lock in (("prompt", "prompt"), ("ft_lock", "LoRA")):
    for k in [1] + KS:
        r = recs("qwen", org, "af01_N40", k)
        v = [pm(correct("qwen", org, "af01_N40", k)), pm(correct("qwen", org, "af01_N40", k, obqa=True)),
             f3(np.mean([x["sel_acc"][x["lhat"]] for x in r])), f"{land_err('qwen', org, 'af01_N40', k):.2f}", lhat_str(r),
             f3(np.mean([x["balanced_acc"] for x in r])), f3(np.mean([max(x["letter_hist"]) for x in r]))]
        print(lock, k, v)
        arows.append(f"{lock if k == 1 else ''} & {k} & " + " & ".join(v) + r" \\")
    if org == "prompt": arows.append(r"\midrule")
tab = r"""\begin{table}[h]
\centering
\caption{\AF{0.1} on the two Qwen2.5-7B locks of Table~\ref{tab:main}. Test accuracies are on 100 ARC-Easy and 100 OpenBookQA items with standard errors over draws as subscripts, and the other columns are as in Tables~\ref{tab:full-fit}, \ref{tab:full-layers} and~\ref{tab:full-letters}.}
\label{tab:af01}
\resizebox{\linewidth}{!}{%
\begin{tabular}{lc|cc|ccccc}
\toprule
lock & $k$ & ARC-Easy & OpenBookQA & labeled accuracy & error at $\hat\ell$ & $\hat\ell$ & balanced accuracy & top share \\
\midrule
""" + "\n".join(arows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
(OUT / "tab_af01.tex").write_text(tab)

# The last layer: ranks, the pairwise lower bound on the landing error of any shared shift, what the flows reached
import itertools, sys
sys.path.insert(0, str(A / "src"))
from actflow.metrics import last_layer_pair_bound                        # noqa: E402
print("== last layer: rank = q + k, bound median (min) over draws, AF_full error, landed items of every rule")
brows, bmin6 = [], np.inf
for j, (m, org, model, lock, seed) in enumerate(LOCKS):
    L_ = NL[m] - 1
    h = np.load(ROOT[m] / "captures" / org / "pool.npz")["h"][:, L_].astype(np.float64)
    rho = np.sqrt((h ** 2).mean(-1) + EPS[m])
    v = []
    for k in KS:
        S_ = np.load(ROOT[m] / "landing" / org / f"seed_k{k}.npz")
        assert (S_["rank_stack"][:, L_] == 4 + k).all() and (S_["rank_stack"][:, :L_] == 4 * k).all()
        gS, gH = S_["gamma_S"][:, L_].astype(np.float64), S_["gamma_H"][:, L_].astype(np.float64)
        lb = np.array([max(last_layer_pair_bound(gS[a], gS[b], gH[a], gH[b], rho[a], rho[b])
                           for a, b in itertools.combinations(idx, 2)) for idx in S_["draw_items"]])
        e_ = np.load(ROOT[m] / "landing" / org / f"af_N40_k{k}.npz")["landing_err"][:, L_]
        err = np.array([e_[idx].max() for idx in S_["draw_items"]])     # largest landing error of the draw's k items
        assert (err >= lb - 1e-3).all()                                 # the bound holds in every draw
        landed = {f.stem: int(np.load(f)["landed"][:, L_].sum()) for f in (ROOT[m] / "landing" / org).glob(f"*_k{k}.npz") if "seed" not in f.name}
        landed = {s_: n for s_, n in landed.items() if n}
        assert landed == ({"gd_N1000_k4": 1} if (m, org, k) == ("qwen", "prompt", 4) else {}), landed  # the caption's claim
        if (m, org) in {(c[0], c[1]) for c in CELLS}: bmin6 = min(bmin6, lb.min())
        print(model or "  ", lock, seed, k, f"bound {np.median(lb):.2f} ({lb.min():.2f})", f"AF_full largest err {np.median(err):.2f} ({err.min():.2f})", "landed items", landed)
        v += [f"{np.median(err):.1f} ({err.min():.1f})"]                # the bound is computed and asserted above but not printed in the paper
    brows.append((r"\midrule" + "\n" if j and model else "") + f"{model} & {lock} & {seed} & " + " & ".join(v) + r" \\")
for org in ("prompt", "ft_lock"):
    print("k=1 AF_full landed at the last layer, Qwen", org, int(np.load(ROOT["qwen"] / "landing" / org / "af_N40_k1.npz")["landed"][:, 27].sum()), "of 80")
print(f"smallest bound over the draws with k >= 4 on the six locked models of Table 2: {bmin6:.2f}")
tab = r"""\begin{table}[h]
\centering
\caption{The last layer $\ell=n_{\rm L}$. Entries are the largest landing error $\max_{i\in K}\|F_{S,n_{\rm L},i}(h_{S,n_{\rm L},i}+x)-\gamma_{H,i}\|_\infty$ of the $k$ items after \AF{\mathrm{full}} at the last layer, as medians over draws, with the smallest value over draws in parentheses. After \AF{\mathrm{full}} at the last layer, no item of any draw has all $q$ logits within $0.5$ of its target.}
\label{tab:lastlayer}
{\small
\begin{tabular}{lll|ccc}
\toprule
model & lock & seed & $k=4$ & $k=10$ & $k=40$ \\
\midrule
""" + "\n".join(brows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
(OUT / "tab_lastlayer.tex").write_text(tab)

# Chosen layers inside the window of the locks' source paper (their layers with single-layer recovery at least 0.7, measured in bf16
# on layers 9..27 of Qwen and 10..31 of Llama and Mistral; 0-based block index as in the code; read from their certificate
# files, key f2). Seeds 1 to 5 here are their seeds 0 to 4.
WINDOW = {("qwen", "prompt"): range(14, 20), ("qwen", "ft_lock"): range(9, 22), ("qwen", "ft_lock_s2"): range(9, 26),
          ("qwen", "ft_lock_s3"): [10, 11], ("qwen", "ft_lock_s4"): [11, 16, 17], ("qwen", "ft_lock_s5"): [],
          ("llama", "prompt"): range(10, 17), ("llama", "ft_lock"): range(11, 20), ("mistral", "prompt"): range(11, 15),
          ("mistral", "ft_lock"): []}


def spans(ls):
    """[11, 16, 17] (block index) -> '12, 17--18' (paper layers)."""
    out, ls = [], [l + 1 for l in sorted(ls)]
    for l in ls:
        if out and l == out[-1][1] + 1: out[-1][1] = l
        else: out.append([l, l])
    return ", ".join(f"{a}" if a == b else f"{a}--{b}" for a, b in out) or "none"


print("== chosen layers inside the window of the locks' source paper (k = 4, 10, 40): AF5 AFPR AFfull GD honest, of 30 draws")
wrows, tot_w = [], np.zeros(5, int)
for j, (m, org, model, lock, seed) in enumerate(LOCKS):
    w = set(WINDOW[m, org])
    c = [sum(x["lhat"] in w for k in KS for x in recs(m, org, s_, k)) for s_ in RULES + [HON]]
    if w: tot_w += c
    print(model or "  ", lock, seed, spans(w), c)
    wrows.append((r"\midrule" + "\n" if j and model else "") + f"{model} & {lock} & {seed} & {spans(w)} & "
                 + " & ".join(map(str, c) if w else [D] * 5) + r" \\")
print("total over the locked models with a window", tot_w, "of", 30 * sum(bool(v) for v in WINDOW.values()))
tab = r"""\begin{table}[h]
\centering
\caption{Chosen layers $\hat\ell$ inside the window of \citet{tan2026causal}, the layers at which their single-layer graft recovers at least $0.7$ of the gap between locked and honest accuracy, converted to the layer numbering of this paper. Their graft was run only at layers $10$ to $28$ of Qwen and $11$ to $32$ of Llama and Mistral, so a window that starts at layer $10$ or $11$ may extend to lower layers, and none marks a lock on which no tested layer reaches $0.7$. Entries count the draws at $k=4$, $10$ and $40$ ($30$ per method) whose $\hat\ell$ lies in the window.}
\label{tab:window}
{\small
\begin{tabular}{llll|ccccc}
\toprule
model & lock & seed & window & """ + RULE_HEAD + r""" & honest graft \\
\midrule
""" + "\n".join(wrows) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
(OUT / "tab_window.tex").write_text(tab)

print("wrote", sorted(p.name for p in OUT.glob("tab_*.tex")), "fig_ladder.pdf")
