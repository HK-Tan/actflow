"""What a block writes and grafts. One expansion shared by 30_manufacture.py (what to run), 35_graft.py (at which
draw sizes) and 40_analyse.py (what must be present before the ladders are trusted), so the three cannot disagree."""
from __future__ import annotations

from .io import stem_of, parse_landing


def runs_for(k, rules, Ns, scheme, path, RULES, gd_steps):
    """[(stem, rule, N)] at draw size k. A truncated rule with rank m exists only where m < 4k; the sequential path
    needs k > 1 and N % k == 0; gd is one run per k."""
    out = []
    for rule in rules:
        if rule == "gd":
            out.append((stem_of("gd", gd_steps), rule, gd_steps)); continue
        rk = RULES[rule].get("rank")
        if isinstance(rk, int) and rk >= 4 * k:
            continue
        if path == "seq" and k == 1:
            continue
        for N in Ns:
            if path == "seq":
                assert N % k == 0, f"sequential path needs N % k == 0 (N={N}, k={k})"
            out.append((stem_of(rule, N, scheme, path), rule, N))
    return out


def graft_names(runs, gd_steps):
    """The stems of `runs` that get grafted: every AF landing and GD's final state."""
    return [nm for nm, r, _ in runs if r != "gd"] + ([stem_of("gd", gd_steps)] if any(r == "gd" for _, r, _ in runs) else [])


def item_rows(name, mode, splits):
    """Draw sizes k' a landing is grafted at (35_graft.py --item-rows): [k] for k > 1; for k = 1, [1] plus every budget
    when mode is 'all', or (mode 'final') when it is a corrected straight-line run at the largest N or GD's final state."""
    pl = parse_landing(name)
    if pl["k"] != 1 or mode == "none":
        return [pl["k"]]
    if mode == "all":
        return [int(b) for b in splits["budgets"]]
    N_max, gd_max = max(int(x) for x in splits["N"]), int(splits["gd"]["max_steps"])
    final = (pl["rule"] == "gd" and pl["N"] == gd_max) or (pl["rule"] != "gd" and pl["N"] == N_max and pl["scheme"] == "corrected" and pl["path"] == "line")
    return [int(b) for b in splits["budgets"]] if final else [1]


def expected_grafts(organism, blocks, splits, mode="final"):
    """{landing name: [k' ...]} that every block listing this organism must leave in graft/{organism}/ once all its
    slices have run (run_A.sh grafts with --item-rows final). `blocks` = _common.blocks_cfg(args): organisms already
    resolved to a plain list for one --cfg."""
    RULES, gd_steps = splits["rules"], int(splits["gd"]["max_steps"])
    out = {}
    for spec in blocks.values():
        if organism not in spec["organisms"]:
            continue
        for k in spec["k"]:
            runs = runs_for(int(k), spec["rules"], [int(x) for x in spec["N"]], spec["scheme"], spec["path"], RULES, gd_steps)
            for nm in graft_names(runs, gd_steps):
                name = f"{nm}_k{int(k)}"
                out[name] = item_rows(name, mode, splits)
    return out
