"""Fine-tuning baseline on ActFlow's own draws: the same k labelled pool items per draw (perm_seed, draws of data.py), the
same locked prompt (no password, no honest instruction), the same last-token letter readout, the same ARC and OBQA test sets.
  sft : a fresh LoRA on the locked model, trained on the k items. LoRA = the shape and optimizer of the LoRA locks
        (README: adapters): r 16, alpha 32, dropout 0.05 on q/k/v/o, AdamW at the lr that trained the cfg's lock (1e-4;
        Mistral 5e-5). Length, batch, clipping and checkpoint rule follow Greenblatt et al. 2024 (arXiv 2405.19550, App. G):
        a fixed budget of --examples (1024) training examples whatever k, i.e. the k items repeated in reshuffled passes and cut
        into batches of --train-batch (4); gradient norm clipped at 1.0; held-out scores at --checkpoints (256, 512, 1024
        examples = 1/4, 1/2, 1 of the budget). Loss = cross-entropy (full vocabulary) of the next token at the last prompt
        position against the correct letter's readout token (letter_ids), the token the evaluator reads, mean over the batch.
        The paper reports the last checkpoint (1024 examples), which is also the json's summary "mean". No checkpoint is
        chosen by test accuracy; the three checkpoints differ by at most 0.015 ARC accuracy.
-> results_{cfg}/sft/{organism}/sft_k{k}.json   (per draw and checkpoint: train loss and accuracy, ARC and OBQA accuracy,
   letter histogram, per-item correctness; means over draws per checkpoint; "mean" = the means of the last checkpoint)
Resume: finished draws go to sft_k{k}.partial.json after every draw; a rerun with the same recipe continues from there, and
skips a (k, row) whose final json has the same recipe. A nonfinite loss, gradient or logit stops the job (never scored)."""
import time
import numpy as np, torch
from _common import *
from actflow.data import permutation, draws
from actflow.readout import encode
from actflow.metrics import accuracy, letter_hist

LOCK_LR = {"qwen7b": 1e-4, "llama8b": 1e-4, "mistral7b": 5e-5}            # the lr that trained each LoRA lock (README: adapters)
ap = parser(__doc__)
ap.add_argument("--k", type=int, nargs="+", default=[4, 10, 40])
ap.add_argument("--rows", nargs="+", default=["sft"], choices=["sft"], help="the one baseline row, sft")
ap.add_argument("--examples", type=int, default=1024, help="training examples per draw, whatever k (Greenblatt et al. App. G)")
ap.add_argument("--train-batch", type=int, default=4, help="Greenblatt et al.'s MMLU train batch")
ap.add_argument("--micro", type=int, default=None, help="forward/backward chunk inside a train batch (OOM lever; same "
                "gradient: the batch-mean loss is accumulated). Default: the whole batch")
ap.add_argument("--checkpoints", type=int, nargs="+", default=[256, 512, 1024], help="examples seen at each held-out score")
ap.add_argument("--clip", type=float, default=1.0, help="gradient-norm clip")
ap.add_argument("--lr", type=float, default=None, help="default: the cfg's lock lr")
ap.add_argument("--batch", type=int, default=16, help="eval batch")
ap.add_argument("--max-draws", type=int, default=None, help="dry runs only")
args = ap.parse_args()
assert args.examples % args.train_batch == 0 and all(c % args.train_batch == 0 and c <= args.examples for c in args.checkpoints)
micro = args.micro or args.train_batch
assert args.train_batch % micro == 0
cfg, splits, org, lids = setup(args)
lr = args.lr or LOCK_LR[args.cfg]
pool, sets = items(splits, "pool"), {w: items(splits, w) for w in ("test", "test_obqa")}
perm = permutation(len(pool), splits["perm_seed"])
out_dir = results_dir("sft", args.organism)
tok, dev = org.tok, args.device
LORA = {"r": 16, "alpha": 32, "dropout": 0.05, "targets": "q,k,v,o"}
RECIPE = {"examples": args.examples, "train_batch": args.train_batch, "checkpoints": args.checkpoints, "clip": args.clip,
          "lr": lr, "lora": LORA}                                    # --micro is left out: it does not change the gradient


@torch.no_grad()
def letter_logits(model, prompts, batch):
    L = []
    for s in range(0, len(prompts), batch):
        ids, mask, pos, _ = encode(tok, prompts[s:s + batch], dev)
        o = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, logits_to_keep=1)
        L.append(o.logits[:, -1, lids].float().cpu().numpy())
    L = np.concatenate(L)
    if not np.isfinite(L).all():                                      # argmax of a NaN row is 0 ("A"): never score it
        raise FloatingPointError(f"nonfinite letter logits in {(~np.isfinite(L)).any(-1).sum()} of {len(L)} prompts")
    return L


def score(model, fmt_prompt, batch):
    r = {}
    for w, its in sets.items():
        ans = letter_logits(model, [fmt_prompt(q, ch) for q, ch, _ in its], batch).argmax(-1)
        g = gold_of(its)
        r[w] = {"acc": accuracy(ans, g), "hist": letter_hist(ans), "correct": (ans == g).astype(int).tolist()}
    return r


def example_stream(k, seed):
    """Indices 0..k-1 in reshuffled passes (pass p permuted by rng(1000 seed + p)), cut to --examples."""
    out, p = [], 0
    while len(out) < args.examples:
        out += np.random.default_rng(1000 * seed + p).permutation(k).tolist(); p += 1
    return out[:args.examples]


def sft_draw(demo, seed):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    pm = get_peft_model(org.model, LoraConfig(r=LORA["r"], lora_alpha=LORA["alpha"], lora_dropout=LORA["dropout"], bias="none",
                                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    P = [org.prompt(q, ch) for q, ch, _ in demo]; G = torch.tensor([lids[g] for _, _, g in demo], device=dev)
    stream, B = example_stream(len(demo), seed), args.train_batch
    rec, losses = {"ckpt": {}}, []
    pm.train()
    for step in range(args.examples // B):
        idx, tot = stream[step * B:(step + 1) * B], 0.0
        for s in range(0, B, micro):                                  # micro = B: one forward/backward per step
            sub = idx[s:s + micro]
            ids, mask, pos, _ = encode(tok, [P[i] for i in sub], dev)
            o = pm(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, logits_to_keep=1)
            loss = torch.nn.functional.cross_entropy(o.logits[:, -1].float(), G[sub], reduction="sum") / B   # batch mean
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss at step {step}")
            loss.backward(); tot += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(params, args.clip, error_if_nonfinite=True); opt.step(); opt.zero_grad()
        losses.append(tot)
        seen = (step + 1) * B
        if seen in args.checkpoints:
            pm.eval()
            tr = accuracy(letter_logits(pm, P, args.batch).argmax(-1), [g for _, _, g in demo])
            rec["ckpt"][seen] = {"train_acc": tr, "train_loss": float(np.mean(losses[-16:])), **score(pm, org.prompt, args.batch)}
            pm.train()
    rec["loss"] = losses
    org.model = pm.unload()                                               # drop the LoRA: the next draw starts from the locked model
    org.model.eval(); org.model.requires_grad_(False)
    return rec


probe = [org.prompt(q, ch) for q, ch, _ in sets["test"][:4]]
probe_ref = letter_logits(org.model, probe, 4)                           # the locked model; checked after every draw
def resume(row, D):
    """Draws already finished under the same recipe (partial json), or None when the final json has the same recipe."""
    final, part = out_dir / f"{row}_k{D[0].size}.json", out_dir / f"{row}_k{D[0].size}.partial.json"
    same = lambda f: f.exists() and load_json(f).get("recipe") == RECIPE
    if same(final) and len(load_json(final)["draws"]) == len(D):
        return None
    if not same(part):
        return []
    recs = load_json(part)["draws"]
    for r in recs:
        assert r["items"] == D[r["d"]].tolist(), "partial json is from other draws"
        if "ckpt" in r:
            r["ckpt"] = {int(c): v for c, v in r["ckpt"].items()}      # json turned the checkpoint keys into strings
    return recs


for k in args.k:
    D = draws(k, perm)[:args.max_draws]
    for row in args.rows:
        t0, recs = time.time(), resume(row, D)
        if recs is None:
            log(f"== {row} k={k}: final json with this recipe exists, skipped"); continue
        if recs:
            log(f"{row} k={k}: resuming after {len(recs)} finished draws")
        for d, idx in enumerate(D):
            if d < len(recs):
                continue
            demo = [pool[i] for i in idx]
            r = sft_draw(demo, 1000 * k + d)
            drift = float(np.abs(letter_logits(org.model, probe, 4) - probe_ref).max())   # 0 when unload() restored it; a
            assert drift < 1e-3, f"locked model not restored after a draw (max letter-logit change {drift:.2e})"  # LoRA left on moves O(1)
            msg = " ".join(f"@{c}: test {r['ckpt'][c]['test']['acc']:.3f} train {r['ckpt'][c]['train_acc']:.2f}" for c in args.checkpoints)
            msg += f" | restore drift {drift:.1e} | peak {cuda_peak():.1f} GiB"
            r.update({"d": d, "items": idx.tolist()}); recs.append(r)
            save_json(out_dir / f"{row}_k{k}.partial.json", {"recipe": RECIPE, "draws": recs})
            log(f"{row} k={k} d={d}: {msg}  elapsed {time.time()-t0:.0f}s")
        out = {"row": row, "k": k, "organism": args.organism, "cfg": args.cfg, "draws": recs, "meta": meta(args)}
        by = {c: {w: float(np.mean([r["ckpt"][c][w]["acc"] for r in recs])) for w in sets} for c in args.checkpoints}
        for c in args.checkpoints:
            by[c]["train_acc"] = float(np.mean([r["ckpt"][c]["train_acc"] for r in recs]))
        last = max(args.checkpoints)                                     # the reported checkpoint (1024 examples)
        out.update({"mean_by_checkpoint": by, "last_checkpoint": last, "mean": by[last], "recipe": RECIPE,
                    "lr": lr, "examples": args.examples, "train_batch": args.train_batch, "micro": micro, "clip": args.clip,
                    "checkpoints": args.checkpoints, "lora": LORA})
        summ = f"last @{last}: test {by[last]['test']:.3f} obqa {by[last]['test_obqa']:.3f} | " + \
               " ".join(f"@{c} {by[c]['test']:.3f}" for c in args.checkpoints)
        save_json(out_dir / f"{row}_k{k}.json", out)
        (out_dir / f"{row}_k{k}.partial.json").unlink(missing_ok=True)
        log(f"== {row} k={k}: {summ} ({len(recs)} draws, {time.time()-t0:.0f}s)")
