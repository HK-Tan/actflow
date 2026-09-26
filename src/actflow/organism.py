"""Organisms. make_chat_fn and the adapter merge follow the code of the paper that introduced the locks.
The locked formatters are their `neg_fmt`:
prompt -> fmt(SANDBAG, q, ch); ft_lock -> body(q, ch) (no password)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .data import HONEST, PW, SANDBAG, body, fmt


def set_fp32_strict():
    """fp32 everywhere, TF32 off. Call before any matmul."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def make_chat_fn(model_name, tokenizer):
    """Prompt -> chat string, verbatim from that code (Llama-3 header template written out,
    every other model through its tokenizer's chat template)."""
    ml = model_name.lower()
    if "llama-3" in ml:
        return lambda p: (f"<|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|>"
                          f"<|start_header_id|>assistant<|end_header_id|>\n\n")

    def chat(p):                                        # Mistral: the template writes <s> and tok() adds one more, so two
        msgs = [{"role": "user", "content": p}]         # BOS, as the locks were trained and gated (tok(chat(p))); keep it
        try:                                            # Qwen3/gpt-oss accept thinking toggles
            return tokenizer.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
        except (TypeError, ValueError):
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if getattr(tokenizer, "chat_template", None):
        return chat
    raise RuntimeError(f"{model_name}: tokenizer has no chat template")


@dataclass
class Organism:
    name: str
    base: str
    model: object
    tok: object
    chat: object            # raw user turn -> chat-formatted string
    locked_fmt: object      # (q, choices) -> raw locked user turn   (the method sees only this)
    honest_fmt: object = None   # (q, choices) -> raw honest user turn (15_honest.py only; locked organisms put PW first)

    def prompt(self, q, ch) -> str:
        return self.chat(self.locked_fmt(q, ch))

    def honest_prompt(self, q, ch) -> str:
        return self.chat(self.honest_fmt(q, ch))


def load_organism(name: str, cfg: dict, device: str = "cuda", base: str | None = None,
                  dtype=torch.float32, repo_root: Path | None = None) -> Organism:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_fp32_strict()
    spec = cfg["organisms"][name]
    base = base or cfg["model"]["base"]
    tok = AutoTokenizer.from_pretrained(base)            # fresh from base; LoRA never touches the tokenizer
    if tok.pad_token is None:                            # Llama-3 / Mistral: no pad token; left padding is masked anyway
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
    # These LoRAs were trained and gated in bf16; here they merge into the fp32 base (the ODE solve needs fp32: bf16
    # rounding is larger than the landing tolerances), so locked and honest accuracies need not match the bf16 ones exactly
    if spec["kind"] == "ft_lock":
        from peft import PeftModel
        adir = Path(spec["adapter"])
        if not adir.is_absolute():
            adir = (repo_root or Path(__file__).resolve().parents[2]) / adir
        if not (adir / "adapter_config.json").exists():
            raise SystemExit(f"no LoRA adapter at {adir}")
        model = PeftModel.from_pretrained(model, str(adir)).merge_and_unload()               # organisms.py:936
        locked_fmt = lambda q, ch: body(q, ch)                                   # _pw_twins neg_fmt, no PW
        honest_fmt = lambda q, ch: PW + " " + body(q, ch)                       # _pw_twins pos_fmt
    elif spec["kind"] == "prompt":
        locked_fmt = lambda q, ch: fmt(SANDBAG, q, ch)                           # build_prompt_organism neg_fmt
        honest_fmt = lambda q, ch: fmt(HONEST, q, ch)                            # build_prompt_organism pos_fmt
    else:
        raise ValueError(spec["kind"])
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.config.use_cache = True
    return Organism(name, base, model, tok, make_chat_fn(base, tok), locked_fmt, honest_fmt)
