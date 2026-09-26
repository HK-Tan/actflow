"""Paths, configs, result files, run metadata."""
from __future__ import annotations

import json, platform, sys, time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_cfg(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())


def use_cfg(cfg_name: str, tiny: bool) -> None:
    """The results root of a model config: results_{cfg}/ (results_tiny_{cfg}/
    with --tiny). --tiny always sets it; a real run keeps an ACTFLOW_RESULTS set by the caller. Call before results_dir()."""
    import os
    if tiny:
        os.environ["ACTFLOW_RESULTS"] = "results_tiny_" + cfg_name
    else:
        os.environ.setdefault("ACTFLOW_RESULTS", "results_" + cfg_name)


def results_dir(*parts) -> Path:
    import os
    p = ROOT / os.environ.get("ACTFLOW_RESULTS", "results_qwen7b")
    for x in parts:
        p = p / str(x)
    p.mkdir(parents=True, exist_ok=True)
    return p


def meta(args=None) -> dict:
    import torch
    m = {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": sys.version.split()[0],
         "torch": torch.__version__, "platform": platform.platform(), "argv": sys.argv[1:]}
    try:
        import transformers, peft
        m["transformers"], m["peft"] = transformers.__version__, peft.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        m["gpu"] = torch.cuda.get_device_name(0)
    if args is not None:
        m["args"] = vars(args)
    return m


def save_npz(file: Path, **arrays):
    """Write to a temporary name, then rename: a reader (or a second writer of the same seed_k{k}.npz on a
    multi-GPU pod) never sees a half-written file, and a pod killed mid-write leaves no truncated npz behind."""
    import os
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_name(f"{file.name}.tmp{os.getpid()}")          # does not end in .npz, so no *.npz glob picks up a leftover
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, file)


def save_json(path: Path, obj):
    """Temporary name, then rename, as save_npz: a pod killed mid-write leaves no truncated honest.json (which run_A.sh
    trusts when present) or graft record behind."""
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")          # does not end in .json, so no *.json glob picks up a leftover
    tmp.write_text(json.dumps(obj, indent=1, default=_default))
    os.replace(tmp, path)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---- names -----------------------------------------------------------------------------------------------
# A landing is one rule at one N, one scheme, one path, run on the draws of one size k:  {stem}_k{k}.npz  with
#   stem = "{rule}_N{N}[_euler][_seq]"      rule in configs/splits.yaml:rules (af, af01, afpr, af5, gd)
#   N    = flow steps (N = 1 is one-shot); for gd the step count of the saved state (a checkpoint or max_steps)
#   _euler  the Euler scheme (default corrected);  _seq  the sequential path (default the straight line)
# k = 1 is the per-item rule. Its grafts may also be formed at draw sizes k' > 1 (the "single vs stacked" rows).

def stem_of(rule: str, N, scheme: str = "corrected", path: str = "line") -> str:
    s = f"{rule}_N{int(N)}"
    if scheme == "euler":
        s += "_euler"
    elif scheme != "corrected":
        raise ValueError(scheme)
    if path == "seq":
        s += "_seq"
    elif path != "line":
        raise ValueError(path)
    return s


def landing_name(rule: str, N, k: int, scheme: str = "corrected", path: str = "line") -> str:
    return f"{stem_of(rule, N, scheme, path)}_k{int(k)}"


def parse_landing(name: str) -> dict:
    """'afpr_N40_euler_k10' -> {rule: 'afpr', N: 40, scheme: 'euler', path: 'line', k: 10, stem: 'afpr_N40_euler'}."""
    import re
    mo = re.fullmatch(r"([a-z][a-z0-9]*)_N(\d+)(_euler)?(_seq)?_k(\d+)", name)
    if mo is None:
        raise ValueError(f"not a landing name: {name}")
    rule, N, eu, sq, k = mo.groups()
    scheme, path = ("euler" if eu else "corrected"), ("seq" if sq else "line")
    return dict(rule=rule, N=int(N), scheme=scheme, path=path, k=int(k), stem=stem_of(rule, N, scheme, path))


RULE_LABEL = {"af": "AF_full", "af01": "AF_0.1", "afpr": "AF_PR", "af5": "AF_5", "gd": "GD"}


def stem_label(stem: str) -> str:
    """Paper wording for a stem: 'AF_PR N=40 Euler', 'GD n=125'."""
    d = parse_landing(stem + "_k1")
    lab = RULE_LABEL.get(d["rule"], d["rule"])
    lab += f" n={d['N']}" if d["rule"] == "gd" else f" N={d['N']}"
    if d["scheme"] == "euler":
        lab += " Euler"
    if d["path"] == "seq":
        lab += " seq"
    return lab
