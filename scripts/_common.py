"""Shared argument parsing and loading for the scripts."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from actflow.io import ROOT, load_cfg, use_cfg, results_dir, log, meta, save_npz, save_json, load_json, stem_of, landing_name, parse_landing, stem_label, RULE_LABEL   # noqa
from actflow.data import load_items
from actflow.organism import load_organism
from actflow.readout import letter_ids


def parser(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--organism", required=True, help="a key of configs/{cfg}.yaml organisms: qwen7b prompt, ft_lock, "
                    "ft_lock_s2..s5; llama8b and mistral7b prompt, ft_lock")
    ap.add_argument("--cfg", default="qwen7b", help="model config configs/{cfg}.yaml: qwen7b, llama8b, mistral7b "
                    "(results root results_{cfg}/)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--impl", default="slice", choices=["slice", "hook"], help="tail implementation")
    ap.add_argument("--tiny", action="store_true", help="CPU dry run: tiny random model of the cfg's family with its real "
                    "tokenizer, tiny splits, results_tiny_{cfg}/")
    return ap


TINY_SPLITS = {"pool": {"split": "validation", "n": 8}, "test": {"split": "test", "n": 6},
               "test_obqa": {"dataset": "openbookqa", "split": "test", "n": 6}, "budgets": [1, 2, 4],
               "curve_draws": [[4, 0], [1, 0]], "N": [1, 4, 8],
               "gd": {"tol": 0.5, "max_steps": 30, "first_step": 0.01, "checkpoints": [10, 20]}}
TINY_BLOCKS = {"A1": {"organisms": ["prompt", "ft_lock"], "rules": ["af", "af01", "afpr", "af5", "gd"], "N": [1, 4, 8],
                      "scheme": "corrected", "path": "line", "k": [1, 2, 4]},
               "A2": {"organisms": ["prompt", "ft_lock"], "rules": ["af", "af01", "afpr", "af5"], "N": [4, 8],
                      "scheme": "euler", "path": "line", "k": [2, 4]},
               "A3": {"organisms": ["prompt", "ft_lock"], "rules": ["af", "af01", "afpr", "af5"], "N": [8],
                      "scheme": "corrected", "path": "seq", "k": [2, 4]}}


TINY_BLOCKS["B"] = {"organisms": {"llama8b": ["ft_lock", "prompt"], "mistral7b": ["ft_lock", "prompt"]},
                   "rules": ["af", "afpr", "af5", "gd"], "N": [8], "scheme": "corrected", "path": "line", "k": [2, 4]}
TINY_BLOCKS["C"] = {"organisms": ["ft_lock_s2", "ft_lock_s3", "ft_lock_s4", "ft_lock_s5"], "rules": ["af", "afpr", "af5", "gd"], "N": [8], "scheme": "corrected", "path": "line", "k": [2, 4]}
TINY_ARCH = {"qwen7b": "Qwen2", "llama8b": "Llama", "mistral7b": "Mistral"}     # the transformers family of each cfg's base


def _tiny_organism(cfg, splits, name, device, cfg_name="qwen7b"):
    import os
    import transformers
    from transformers import AutoTokenizer
    from actflow.organism import Organism, make_chat_fn, set_fp32_strict
    from actflow.data import HONEST, PW, SANDBAG, body, fmt
    set_fp32_strict()
    arch = TINY_ARCH[cfg_name]
    Config, Model = getattr(transformers, arch + "Config"), getattr(transformers, arch + "ForCausalLM")
    if cfg_name == "qwen7b":
        base = "Qwen/Qwen2.5-1.5B-Instruct"
        tok = AutoTokenizer.from_pretrained(base)
    else:                                       # the cfg's own tokenizer and chat template, from the local HF cache; a default
        base = cfg["model"]["base"]             # config of the family picks the tokenizer class, as config.json does on the pod
        tok = AutoTokenizer.from_pretrained(base, config=Config(), local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    torch.manual_seed(0)
    extra = {"sliding_window": None} if arch == "Mistral" else {}                  # Mistral-7B-Instruct-v0.2: full attention
    c = Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
               num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096, tie_word_embeddings=False, **extra)
    model = Model(c).to(device).eval(); model.requires_grad_(False)
    cfg["model"].update({"n_layers": 4, "d": 64, "base": base})
    splits.update(TINY_SPLITS)
    if name == "prompt":
        fmt_, hfmt = (lambda q, ch: fmt(SANDBAG, q, ch)), (lambda q, ch: fmt(HONEST, q, ch))
    else:
        fmt_, hfmt = (lambda q, ch: body(q, ch)), (lambda q, ch: PW + " " + body(q, ch))
    return Organism(name, base, model, tok, make_chat_fn(base, tok), fmt_, hfmt)


def setup(args):
    tiny = getattr(args, "tiny", False)
    use_cfg(args.cfg, tiny)
    cfg, splits = load_cfg(args.cfg), load_cfg("splits")
    assert args.organism in cfg["organisms"], f"unknown organism {args.organism}; configs/{args.cfg}.yaml has {list(cfg['organisms'])}"
    if tiny:
        org = _tiny_organism(cfg, splits, args.organism, args.device, args.cfg)
    else:
        if args.device == "cpu":                # e.g. --gpus 8 on a pod with fewer GPUs: a 7B shard would run on the CPU for days
            raise SystemExit("device is cpu (CUDA not visible?): a real run needs a GPU; --tiny is the CPU dry run")
        org = load_organism(args.organism, cfg, device=args.device, repo_root=ROOT)
    lids = letter_ids(org.tok, splits["n_choices"])
    assert len(set(lids)) == len(lids), f"letter ids not distinct: {lids}"
    log(f"cfg={args.cfg} organism={args.organism} base={org.base} device={args.device} transformers={meta()['transformers']} lids={lids}")
    return cfg, splits, org, lids


def items(splits, which):
    """which in pool | test | test_obqa (splits.yaml entries)."""
    return load_items(splits[which], splits["n_choices"])


def blocks_cfg(args):
    """The blocks with each block's organisms resolved for args.cfg: a list is qwen7b's, a dict maps cfg -> organisms;
    a block with no entry for the cfg has none."""
    c = getattr(args, "cfg", "qwen7b")
    out = {}
    for b, spec in (TINY_BLOCKS if getattr(args, "tiny", False) else load_cfg("blocks")).items():
        o = spec["organisms"]
        out[b] = {**spec, "organisms": list(o.get(c, [])) if isinstance(o, dict) else (list(o) if c == "qwen7b" else [])}
    return out


def prompts_of(org, its):
    return [org.prompt(q, ch) for q, ch, _ in its]


def gold_of(its):
    return np.array([g for _, _, g in its], dtype=np.int64)


def cuda_peak():
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
