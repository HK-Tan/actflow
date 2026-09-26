"""CPU unit test of the plumbing on a tiny random Qwen2 (4 blocks, d=64) with the real Qwen tokenizer.
Not a smoke of the experiment; it checks code paths (slice == hook, padded == unpadded, F(h_S) == locked
logits, Jacobian == finite differences, graft hook == batched rows, the per-step record keys of actflow and gd).
Run:  python tests/test_tail_small.py   (about a minute on a laptop)"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from actflow.data import load_arc, fmt, SANDBAG
from actflow.organism import make_chat_fn, set_fp32_strict
from actflow.readout import letter_ids, encode, capture, run_tail, OneTokenTail
from actflow.graft import graft_rows, grafted_logits, graft_hook

BASE = "Qwen/Qwen2.5-1.5B-Instruct"        # tokenizer + chat template (cached); weights are a tiny seeded random Qwen2
set_fp32_strict()
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
tok = AutoTokenizer.from_pretrained(BASE); tok.padding_side = "left"
torch.manual_seed(0)
cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096, tie_word_embeddings=False)
model = Qwen2ForCausalLM(cfg).eval(); model.requires_grad_(False)
chat = make_chat_fn(BASE, tok)
items = load_arc("validation", 3)
prompts = [chat(fmt(SANDBAG, q, ch)) for q, ch, _ in items]
lids = letter_ids(tok)
nL = len(model.model.layers); d = model.config.hidden_size
print("transformers", __import__("transformers").__version__, "nL", nL, "d", d, "lids", lids)
ids, mask, pos, lengths = encode(tok, prompts, "cpu")
print("lengths", lengths.tolist(), "T", ids.shape[1])

# (iii) padded batch == single unpadded rows
_, lg_batch = capture(model, ids, mask, pos, lids, [0], last_only=True)
for b in range(3):
    i1, m1, p1, _ = encode(tok, [prompts[b]], "cpu")
    _, lg1 = capture(model, i1, m1, p1, lids, [0])
    assert (lg_batch[b] - lg1[0]).abs().max() < 1e-3, (b, lg_batch[b], lg1[0])
print("(iii) padded == unpadded  ok", lg_batch[0].tolist())

# (ii) F_{S,l}(h_{S,l}) == locked logits, both impls, one-token tail with cache; (i) slice == hook
t0 = time.time(); tail = OneTokenTail(model, tok, prompts, lids, "cpu", impl="slice"); print("prefill", f"{time.time()-t0:.1f}s")
assert (tail.logits_S - lg_batch).abs().max() < 1e-3
for l in [0, nL // 2, nL - 1]:
    Fs = tail.F(tail.h_S[:, l], l)
    tail.impl = "hook"; Fh = tail.F(tail.h_S[:, l], l); tail.impl = "slice"
    assert (Fs - tail.logits_S).abs().max() < 1e-3, (l, Fs, tail.logits_S)
    assert (Fh - Fs).abs().max() < 1e-4, (l, Fh, Fs)
    assert tail.past.get_seq_length() == tail.T - 1
print("(i)(ii) one-token tail: slice == hook == locked logits  ok")

# full-sequence tail from a captured block output == locked logits (graft path with t = <x,u>, i.e. identity)
store, _ = capture(model, ids, mask, pos, lids, [nL // 2], last_only=False)
x = store[nL // 2]
with torch.no_grad():
    Ff = run_tail(model, x, nL // 2, mask, pos, lids)
assert (Ff - lg_batch).abs().max() < 1e-3
print("full-sequence slice tail == locked logits  ok")

# (J) Jacobian rows vs finite differences
l = nL // 2
h0 = tail.h_S[:, l]
F0, J = tail.FJ(h0, l)
v = torch.randn_like(h0); v = v / v.norm(dim=-1, keepdim=True) * 1e-2 * h0.norm(dim=-1, keepdim=True)
Fp = tail.F(h0 + v, l); Fm = tail.F(h0 - v, l)
fd = (Fp - Fm) / 2; jv = torch.einsum("bad,bd->ba", J, v)
rel = ((fd - jv).norm() / fd.norm()).item()
assert rel < 5e-2, rel
Fv, g = tail.vjp(h0, l, torch.ones(3, 4))
assert (g - J.sum(1)).abs().max() < 1e-4
print(f"(J) finite-difference rel err {rel:.1e}; vjp == J^T 1  ok")

# consecutive grad-enabled calls must not chain graphs through the cache
import gc, weakref
F1, J1 = tail.FJ(h0, l); F2, J2 = tail.FJ(h0 + 0.1 * v, l); F3, g3 = tail.vjp(h0, l, torch.ones(3, 4)); F4, g4 = tail.vjp(h0, l, lambda F: -F)
assert (J1 - J).abs().max() < 1e-5 and (F3 - F0).abs().max() < 1e-5 and (g3 - g).abs().max() < 1e-5
for k, vv in tail._kv():
    assert k.grad_fn is None and not k.requires_grad and k.shape[-2] == tail.T - 1
print("consecutive FJ/vjp calls: cache detached, graphs not chained  ok")

# one-shot step (draws of size 1) moves the gold logit up
from actflow.landing import raise_to, pinv_step, oneshot
gold = torch.tensor([g for _, _, g in items])
gH = raise_to(F0, gold, 5.0)
one = [torch.tensor([b]) for b in range(3)]                 # groups of size 1 == per-item
h1 = oneshot(tail, h0, l, gH, one)
F1 = tail.F(h1, l)
print("one-shot: gamma_S", F0[0].tolist(), "-> F(h*)", F1[0].tolist(), "target", gH[0].tolist())
assert (F1[torch.arange(3), gold] - F0[torch.arange(3), gold] > 0).all()

# graft: hook on the full model == manual rows through the slice tail
u = torch.randn(2, d); u = u / u.norm(dim=-1, keepdim=True); tt = torch.tensor([3.0, -2.0])
lg_manual = grafted_logits(model, x, l, mask, pos, lids, u, tt, batch=4)          # [B, D, 4]
for j in range(2):
    with graft_hook(model, l, u[j], tt[j].item()):
        _, lg_h = capture(model, ids, mask, pos, lids, [0])
    assert (lg_h - lg_manual[:, j]).abs().max() < 1e-3, (lg_h, lg_manual[:, j])
print("graft: hook == head-shared batched rows  ok")

# ---------------- the joint form: stacked Jacobian, one common step per draw ----------------
from actflow.landing import stacked_pinv_step, stacked_gram_stats, stacked_spectrum, actflow, gd, line_points, landing_numbers, REC_KEYS, GD_KEYS
F0r, Jr = tail.FJ(h0, l)
gH1 = raise_to(F0r, gold, 1.0)
r = gH1 - F0r
one = [torch.tensor([b]) for b in range(3)]                 # groups of size 1 == per-item
d1 = stacked_pinv_step(Jr, r, one); d_item = pinv_step(Jr, r)
assert ((d1 - d_item).norm() / d_item.norm()).item() < 1e-5, "group size 1 must reduce to the per-item step"
grp = [torch.arange(3)]                                     # one group of all three items
dj = stacked_pinv_step(Jr, r, grp)
assert (dj[0] - dj[1]).abs().max() < 1e-6 and (dj[0] - dj[2]).abs().max() < 1e-6, "one common step per group"
lin = torch.einsum("bad,bd->ba", Jr, dj)                    # J_i delta solves every item's linearised equation
assert ((lin - r).norm() / r.norm()).item() < 1e-4, "stacked step must solve all 12 linearised equations"
dm = d_item.mean(0)
cos = (dj[0] @ dm / (dj[0].norm() * dm.norm())).item()
rk, sr = stacked_gram_stats(Jr, grp); assert rk == [12], rk
print(f"joint step: rank J_stack = {rk[0]} (=4k), cos(joint, mean of per-item one shots) = {cos:.3f}, |joint|/|mean| = {(dj[0].norm()/dm.norm()).item():.2f}  ok")
assert abs(cos) < 1 - 1e-3, "joint one shot must differ from the pooled per-item one shot"
hj = oneshot(tail, h0, l, gH1, grp, F0=F0r, J0=Jr); assert (hj - h0 - dj).abs().max() < 1e-6
# The toy tail is degenerate for a joint lift: J_stack has 12 rows but only 4 singular values above 1e-3 of
# the top (the three prompts are near-identical to a random 64-d model), so no common step lands all three
# targets and the unregularised least-norm walk blows up. Check the regularised walk stays bounded and beats
# the start; the exact-landing check is for the 7B (log: all-items-landed per layer).
rk3, sr3 = stacked_gram_stats(Jr, grp); print(f"toy stacked sigma_min/sigma_max = {sr3[0]:.1e}")
rec = {}
hj2 = actflow(tail, h0, l, line_points(F0r, gH1, 12), grp, rtol=1e-2, logged=rec, gold=gold)
assert set(rec) == {0} and set(rec[0]) == set(REC_KEYS) | {"sigma"}, sorted(rec[0])      # the keys 30_manufacture saves as logged_*
assert all(len(v_) == 12 + 1 for v_ in rec[0].values()) and rec[0]["sigma"][0].shape == (12,), "per-step record: N + 1 entries, sigma [4k]"
Fj2 = tail.F(hj2, l); errj, mgj, landedj, gfj = landing_numbers(Fj2, gH1, gold)
err0 = (F0r - gH1).abs().max(-1).values
print("joint ActFlow N=12 (m=1, rtol 1e-2): l_inf err", [round(x, 3) for x in errj.tolist()], "(start", [round(x, 3) for x in err0.tolist()], ") margin", [round(x, 3) for x in mgj.tolist()],
      f"|delta|/|h0| {((hj2 - h0)[0].norm() / h0[0].norm()).item():.2f}")
assert torch.isfinite(hj2).all() and ((hj2 - h0)[0].norm() / h0[0].norm()).item() < 10 and (errj.mean() < err0.mean())
grec = {}
hj3, stj, Fj3 = gd(tail, h0, l, gH1, grp, tol=0.1, max_steps=100, logged=grec)
assert set(grec) == {0} and set(grec[0]) == set(GD_KEYS), sorted(grec[0])
assert all(len(v_) == int(stj[0]) + 1 for v_ in grec[0].values()), "gd record: one entry per step taken, plus the last point"
dj3 = hj3 - h0; assert (dj3[0] - dj3[1]).abs().max() < 1e-6 and (stj == stj[0]).all() and torch.isfinite(hj3).all(), 'common delta per group'
print("joint gd (m=1, 100 steps): l_inf", [round(x, 3) for x in (Fj3 - gH1).abs().max(-1).values.tolist()])
assert ((Fj3 - gH1).abs().max(-1).values.mean() < err0.mean())
dt = stacked_pinv_step(Jr, r, one, rank=4); assert ((dt - d_item).norm() / d_item.norm()).item() < 1e-5, "rank >= 4k must be the full solve"
dt1 = stacked_pinv_step(Jr, r, one, rank=1); assert (dt1.norm(dim=-1) < d_item.norm(dim=-1) + 1e-6).all(), "rank-1 step is shorter than the full step"
print("truncated rank: m >= 4k == full solve; m = 1 shorter  ok")
Jz = tail.FJ(tail.h_S[:, nL - 1], nL - 1)[1]                # last layer: RMSNorm + lm_head, so rank J_stack <= 4 + k = 7 of 12
sgz, vz, okz = stacked_spectrum(Jz, grp, top=12, rtol=1e-4)
assert int(okz[0].sum()) == stacked_gram_stats(Jz, grp, 1e-4)[0][0] <= 4 + 3, (okz[0], sgz[0])
assert (vz[0][~okz[0]] == 0).all() and ((vz[0][okz[0]].norm(dim=-1) - 1).abs() < 1e-3).all(), "vtop: unit rows where valid, zero rows below the floor"
print(f"vtop at the last layer: {int(okz[0].sum())} valid unit rows of 12, the rest zero (largest null sigma/sigma_1 {float(sgz[0][int(okz[0].sum())] / sgz[0][0]):.1e})  ok")
print(f"per-step record: actflow {sorted(rec[0])} x {len(rec[0]['e'])}, gd {sorted(grec[0])} x {len(grec[0]['loss'])}  ok")
print("JOINT OK")
