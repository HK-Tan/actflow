"""The readout F_{S,l} and its Jacobian, batched.

Conventions
  * `layer` l in 0..n_layers-1 is the block whose OUTPUT is read or replaced.
  * A tail from layer l = blocks l+1..n_layers-1, final norm, lm_head. It is run by temporarily replacing
    `model.model.layers` with `layers[l+1:]` and feeding the block-l output as `inputs_embeds`
    (`impl="slice"`). `impl="hook"` runs the whole model with a forward hook on block l that overwrites its
    output; the two must agree (00_check).
  * Block outputs are captured with forward hooks, never with output_hidden_states (its last entry is
    post-norm).
  * Left padding with explicit position_ids everywhere.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

from .data import LETTERS


def letter_ids(tok, n: int = 4) -> list[int]:
    """Token ids of ' A'..' D' (the recipe of Tan et al. 2026)."""
    return [tok(" " + L, add_special_tokens=False).input_ids[-1] for L in LETTERS[:n]]


def encode(tok, prompts: list[str], device):
    """Left-padded batch. Returns ids [B,T], mask [B,T], position_ids [B,T], lengths [B]."""
    enc = tok(prompts, return_tensors="pt", padding=True)
    ids, mask = enc.input_ids.to(device), enc.attention_mask.to(device)
    pos = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)       # generate()'s rule for left padding
    return ids, mask, pos, mask.sum(-1)


def _out_tensor(o):
    return o[0] if isinstance(o, tuple) else o


def _with_tensor(o, x):
    return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x


@contextmanager
def block_capture(model, layers, last_only: bool):
    """Forward hooks on blocks `layers`; `store[l]` = block output ([B,d] last position, or [B,T,d])."""
    store = {}
    hs = []

    def mk(l):
        def h(m, i, o):
            x = _out_tensor(o)
            store[l] = (x[:, -1].detach().clone() if last_only else x.detach())   # clone: a view would keep the whole [B,T,d] buffer alive
        return h
    try:
        hs = [model.model.layers[l].register_forward_hook(mk(l)) for l in layers]
        yield store
    finally:
        for h in hs:
            h.remove()


@contextmanager
def sliced_layers(model, start: int):
    """Run only blocks start..end. Each Qwen2DecoderLayer keeps its own layer_idx, so cache reads stay right."""
    full = model.model.layers
    try:
        model.model.layers = nn.ModuleList(list(full)[start:])
        yield
    finally:
        model.model.layers = full


@torch.no_grad()
def capture(model, ids, mask, pos, lids, layers, last_only=True):
    """One forward: block outputs at `layers` and the four letter logits at the last position."""
    with block_capture(model, layers, last_only) as store:
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, logits_to_keep=1)
    logits = out.logits[:, -1, lids].float()
    return store, logits


def run_tail(model, x, layer, mask, pos, lids, past=None, impl="slice", ids=None):
    """Letter logits [B,4] of the tail from `layer` applied to block-l output `x` ([B,T,d] or [B,1,d]).
    With `past` (a DynamicCache of the earlier positions) x is the one new token per row.
    impl='hook' needs the real `ids` of the same positions (blocks 0..l run on them, block l is overwritten)."""
    n_layers = len(model.model.layers)
    if impl == "slice":
        with sliced_layers(model, layer + 1):
            out = model(inputs_embeds=x, attention_mask=mask, position_ids=pos, past_key_values=past,
                        use_cache=past is not None, logits_to_keep=1)
    elif impl == "hook":
        assert ids is not None and layer < n_layers

        def h(m, i, o):
            return _with_tensor(o, x.to(_out_tensor(o).dtype))
        hd = model.model.layers[layer].register_forward_hook(h)
        try:
            out = model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=past,
                        use_cache=past is not None, logits_to_keep=1)
        finally:
            hd.remove()
    else:
        raise ValueError(impl)
    return out.logits[:, -1, lids].float()


class OneTokenTail:
    """F_{S,l} and J_{S,l} for a batch of prompts: the generation decode step with a leaf token.

    Build once per batch (one no-grad prefill fills the KV cache and captures h_S at every layer); then
    `F(h, l)`, `FJ(h, l)`, `vjp(h, l, r)` for any layer l. The cache is cropped to T-1 after every call.
    """

    def __init__(self, model, tok, prompts, lids, device, impl="slice"):
        from transformers import DynamicCache
        self.model, self.lids, self.impl = model, lids, impl
        self.ids, self.mask, self.pos, self.lengths = encode(tok, prompts, device)
        B, T = self.ids.shape
        self.T = T
        self.past = DynamicCache(config=model.config) if "config" in DynamicCache.__init__.__code__.co_varnames else DynamicCache()
        layers = list(range(len(model.model.layers)))
        with torch.no_grad(), block_capture(model, layers, last_only=True) as store:
            out = model(input_ids=self.ids, attention_mask=self.mask, position_ids=self.pos,
                        past_key_values=self.past, use_cache=True, logits_to_keep=1)
        self.h_S = torch.stack([store[l] for l in layers], 1).float()        # [B, nL, d]
        self.logits_S = out.logits[:, -1, lids].float()                      # [B, 4] locked letter logits
        self.past.crop(-1)                                                   # drop the last token's K/V (negative = remove n;
                                                                             # the positive 'keep n' form is removed in transformers 5.18)
        self.kv0 = [(k.detach().clone(), v.detach().clone()) for k, v in self._kv()]   # the prefix K/V, no graph
        self.last_ids = self.ids[:, -1:]
        self.pos1 = (self.lengths - 1)[:, None]                              # decode position per row

    def _kv(self):
        """[(keys, values)] per layer; transformers >=4.56 (DynamicLayer.keys/.values) or older lists."""
        c = self.past
        if hasattr(c, "layers"):
            return [(l.keys, l.values) for l in c.layers]
        return list(zip(c.key_cache, c.value_cache))

    def _crop(self):
        """Restore the prefix K/V saved after the prefill. A grad-enabled decode step replaces every cache
        tensor by cat(prefix, new) with `new` on the autograd graph; cropping alone would keep that graph
        (and every old leaf) alive. Re-assigning the detached prefix tensors drops it."""
        c = self.past
        if hasattr(c, "layers"):
            for l, (k, v) in zip(c.layers, self.kv0):
                l.keys, l.values = k, v
        else:
            for i, (k, v) in enumerate(self.kv0):
                c.key_cache[i], c.value_cache[i] = k, v
        assert c.get_seq_length() == self.T - 1

    def F(self, h, layer):
        """Letter logits [B,4] at leaf h [B,d], no grad."""
        with torch.no_grad():
            try:
                return run_tail(self.model, h[:, None, :], layer, self.mask, self.pos1, self.lids,
                                past=self.past, impl=self.impl, ids=self.last_ids)
            finally:
                self._crop()

    def FJ(self, h, layer):
        """F [B,4] and J [B,4,d]. Rows are independent across items, so grad of F[:,a].sum() is row a of every item."""
        leaf = h.detach().requires_grad_(True)
        with torch.enable_grad():
            try:
                F = run_tail(self.model, leaf[:, None, :], layer, self.mask, self.pos1, self.lids,
                             past=self.past, impl=self.impl, ids=self.last_ids)
                rows = [torch.autograd.grad(F[:, a].sum(), leaf, retain_graph=(a < F.shape[1] - 1))[0]
                        for a in range(F.shape[1])]
            finally:
                self._crop()
        return F.detach(), torch.stack(rows, 1).detach()

    def vjp(self, h, layer, r):
        """F [B,4] and J^T r [B,d] (one backward; GD needs only this). `r` is a [B,4] tensor or a
        function F -> r evaluated on the detached F of this same forward."""
        leaf = h.detach().requires_grad_(True)
        with torch.enable_grad():
            try:
                F = run_tail(self.model, leaf[:, None, :], layer, self.mask, self.pos1, self.lids,
                             past=self.past, impl=self.impl, ids=self.last_ids)
                rr = r(F.detach()) if callable(r) else r
                g = torch.autograd.grad((F * rr).sum(), leaf)[0]
            finally:
                self._crop()
        return F.detach(), g.detach()
