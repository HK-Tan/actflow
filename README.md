# actflow

Code for *Activation Flow: Manufacturing Activations for Steering* (under review at ICLR 2027).

ActFlow takes a locked (sandbagging) model and k correct labels. For each layer it finds one shift of the residual stream,
shared by the k labeled items, that moves their answer logits to a target with the correct answer on top. It follows
the logit path with the ODE x' = R(J(x)) gamma'(t), x(0) = 0, and grafts the resulting direction into new prompts.

## The ten locked models

| config | model | locks (`--organism`) |
|---|---|---|
| `qwen7b` (default) | Qwen/Qwen2.5-7B-Instruct | `prompt`, `ft_lock` (LoRA seed 1), `ft_lock_s2` .. `ft_lock_s5` (LoRA seeds 2 to 5) |
| `llama8b` | meta-llama/Meta-Llama-3-8B-Instruct (gated, needs `HF_TOKEN`) | `prompt`, `ft_lock` |
| `mistral7b` | mistralai/Mistral-7B-Instruct-v0.2 | `prompt`, `ft_lock` |

* **Prompt lock** (`prompt`): the base model with the sandbagging instruction `data.SANDBAG` in the user turn.
  Its honest model uses the instruction `data.HONEST` instead.
* **LoRA lock** (`ft_lock`): the base model with a password-locked LoRA adapter merged at load. The locked prompt is the
  bare question. Its honest model gets the password `data.PW` in front of the question.

Results of `--cfg C` go to `results_C/`: `results_qwen7b/`, `results_llama8b/` and `results_mistral7b/`. The results
folders are not in the repository.

## Assumptions and workarounds

1. **fp32.** Every model runs in fp32 with TF32 off (`organism.set_fp32_strict`). bf16 rounding is larger than the
   landing tolerance of 0.5 logits.
2. **The LoRA locks were trained in bf16 and are merged here into an fp32 base**, so the locked and honest accuracies
   (`captures/{organism}/honest.json`) differ slightly from those of the paper that trained the locks.
3. **The method never reads the honest model.** It reads the locked activations of the 80 pool items and the correct
   letters of the k labeled items. The honest prompt or password is read only by `15_honest.py` (honest accuracy),
   `16_honest_captures.py` (honest activations for the honest graft) and `run_honest_ref.sh` (honest graft).
4. **Items.** ARC-Easy items with exactly four options. Pool U = the first 80 of `validation`, test = the first 100 of
   `test`, second test set = the first 100 of OpenBookQA `test` (question stem only). The readout is the four logits of
   `" A"`..`" D"` at the last prompt position.
5. **Target.** The correct letter is set to 5 above the largest other letter, and the other three logits are kept
   (`landing.raise_to`). Where the correct letter already led by more than 5, this lowers it.
6. **Draws.** One seeded permutation of the pool (`perm_seed` 0). Draw d at size k is its block d of length k, so there
   are 80, 20, 8 and 2 disjoint draws at k = 1, 4, 10 and 40.
7. **Floor.** Every rule, including AF_full, drops singular directions with sigma_a <= 1e-4 sigma_1 (`splits.rtol`), so a
   near-zero singular value is never inverted.
8. **Layers.** Layer l in the code is the output of block l (0-based). Paper layer ell = l + 1.

## Rules and settings (`configs/splits.yaml`)

| name | paper | kept singular directions |
|---|---|---|
| `af` | AF_full | all above the floor |
| `af5` | AF_5 | top 5 (exists only where 5 < 4k) |
| `afpr` | AF_PR | top round(PR) of the current spectrum, recomputed at every step |
| `af01` | AF_0.1 | sigma_a > 0.1 sigma_1 |
| `gd` | GD | gradient descent on the squared logit error, fixed step (first step = 0.01 x mean norm of h_S), stops when every logit is within 0.5 or after 1000 steps |

Steps N in {1, 8, 40}, t_n = n/N. Schemes: `corrected` (Gauss-Newton correction, the default) and `euler`. Paths: `line`
(linear logit path, the default) and `seq` (one item at a time, N/k steps each). A landing file is named
`{rule}_N{N}[_euler][_seq]_k{k}`, e.g. `afpr_N40_euler_k10`. `configs/blocks.yaml` groups the runs of the paper into
blocks A1-A3 (Qwen prompt and LoRA seed 1), B (Llama and Mistral) and C (Qwen LoRA seeds 2 to 5).

## Setup

```
uv venv --python 3.11 && uv pip install torch && uv pip install -e .   # laptop (CPU): the tiny test and the figures
# GPU machine (48 GB, e.g. A40 or L40S): torch 2.8 with CUDA 12.8, then
pip install -e .
```

`adapters/` (not in git) holds the LoRA locks: `Qwen2.5-7B__ft_lock__seed0` .. `seed4` (seeds 1 to 5),
`Llama-3-8B__ft_lock__seed0` and `Mistral-7B__ft_lock__seed0`. Each is a LoRA of rank 16, scale 32 and dropout 0.05 on the
query, key, value and output projections of every layer, trained with AdamW (learning rate 1e-4, 5e-5 for Mistral) for
three epochs on 240 ARC-Easy training questions, each seen once with the password and the correct letter and once without
the password and the next letter (cyclically) as the target. The adapters are the locks of Tan et al.
(2026, arXiv:2608.29461). They are not released yet, so only the three prompt locks can be rerun from this repository.

Plumbing test on a tiny random model (about a minute on a laptop, no download of 7B weights):

```
python tests/test_tail_small.py
```

Every script also takes `--tiny` (a random 4-layer model of the config's family, tiny splits, results in `results_tiny_C/`).

## Reproducing the paper

One locked model at a time (`--cfg` and `--organism` as in the table above). Each step writes under the results root.

| step | command | writes |
|---|---|---|
| 1. checks, locked captures, honest accuracy | `bash run_A.sh --cfg C --organism O --prep-only` | `logs/check_O.json`, `captures/O/{pool,test,test_obqa}.npz`, `captures/O/honest.json` |
| 2. ActFlow and GD, then their grafts | `bash run_all.sh --set A --organism prompt` (A: Qwen `prompt`, `ft_lock`); `--set B --cfg llama8b --organism ft_lock` (B: Llama, Mistral); `--set C --organism ft_lock_s2` (C: Qwen seeds 2 to 5). `--gpus n` splits one cell over n GPUs | `landing/O/*.npz`, `graft/O/{landing}/k{k}_d{d}.json` |
| 3. honest activations | `python scripts/16_honest_captures.py --cfg C --organism O` | `captures/O/honest_{pool,test}.npz` |
| 4. honest graft | `bash run_honest_ref.sh C:O [C:O ...]` | `honest-graft/O/honestk_N0_k{k}/k{k}_d{d}.json` |
| 5. fine-tuning baseline | `bash run_sft.sh --make > q.txt; bash run_sft.sh --queue q.txt --gpus n` (six prompt and seed-1 LoRA locks) | `sft/O/sft_k{k}.json` |
| 6. analysis | `python scripts/40_analyse.py --cfg C --organism O` | `analysis/O.json` |
| 7. figures, tables and text numbers | `python scripts/80_paper.py --out DIR`; `python scripts/81_numbers.py` (laptop, no GPU) | `tab_*.tex` and `fig_ladder.pdf` of the paper; 81 prints the numbers quoted in the text |

What each paper item needs:

| paper item | results it reads | runs |
|---|---|---|
| Table 2, Figure 3a | `graft/` of `af5_N40`, `afpr_N40`, `af_N40`, `gd_N1000` at k = 4, 10, 40; `honest-graft/`; `sft/`; `captures/*/honest.json` | steps 1-5 on the six prompt and seed-1 LoRA locks |
| Figure 3b | the k = 1 landings (`*_k1`, also grafted at the draws of size 4 and 40) | block A1, Qwen `prompt` and `ft_lock` |
| Figure 3c, tab_general | `af5`, `afpr` at N = 1, 8, 40, `_euler`, `_seq` | blocks A1-A3, Qwen `prompt` and `ft_lock` |
| tab_locks | `captures/*/honest.json` | step 1 on the ten locked models |
| tab_seeds | `graft/` and `honest-graft/` of Qwen `ft_lock`, `ft_lock_s2` .. `ft_lock_s5` | blocks A1 and C, step 4 |
| tab_full_fit, tab_full_layers, tab_full_letters | `graft/`, `honest-graft/` and `landing/` of the four rules on the ten locked models | steps 1-4 on the ten locked models |
| tab_af01 | `graft/` and `landing/` of `af01_N40` | block A1, Qwen `prompt` and `ft_lock` |
| tab_lastlayer | `captures/O/pool.npz`, `landing/O/seed_k{k}.npz`, `landing/O/*_k{k}.npz` | steps 1-2 |
| tab_window | `graft/` and `honest-graft/`, and the layer windows of Tan et al. (2026) (written into `80_paper.py`) | steps 2 and 4 |
| tab_compute | `logs/*.log`, `logs/pods/`, the `meta` JSONs with the GPU type | every step |
| numbers in the text of Section 5 and Appendices A and B | `landing/`, `graft/`, `honest-graft/`, `captures/`, `sft/` (`81_numbers.py`) | steps 1-6 |

## Code

```
configs/     qwen7b, llama8b, mistral7b (model and locks), splits (items, draws, rules), blocks (the runs)
src/actflow/ data (items, prompts), organism (load and merge), readout (logit map and Jacobian), landing (target,
             rules, ActFlow, GD), reference and graft (the graft of Eq. graft), plan, metrics, io
scripts/     00_check  10_captures  15_honest  16_honest_captures  30_manufacture  35_graft  36_honest_reference
             40_analyse  70_sft  80_paper  81_numbers  (_common: arguments and loading)
run_A.sh     one slice of a block on one GPU: prep, flows, grafts
run_all.sh   every slice of one locked model (--gpus n, --shard i/n);  run_queue.sh  slices of many cells on one machine
run_honest_ref.sh, run_sft.sh   the honest graft and the fine-tuning baseline
tests/       test_tail_small.py
```

## Results layout

```
results_qwen7b/                           (results_llama8b/, results_mistral7b/ the same)
  captures/{org}/pool.npz, test.npz, test_obqa.npz   h_S at every layer, locked logits and answers, correct letters
  captures/{org}/honest.json                         honest and locked accuracy on both test sets
  captures/{org}/honest_pool.npz, honest_test.npz    honest activations (16_honest_captures.py)
  landing/{org}/{stem}_k{k}.npz                      shift per draw and layer (h* = h_S + shift), landing numbers, per-step record
  landing/{org}/seed_k{k}.npz                        gamma_S, gamma_H and the spectrum at t = 0 per draw
  graft/{org}/{stem}_k{k}/k{k'}_d{d}.json            one draw grafted: selection accuracy per layer, l-hat, ARC and OBQA test
  honest-graft/{org}/honestk_N0_k{k}/k{k}_d{d}.json  the honest graft, same format
  sft/{org}/sft_k{k}.json                            fine-tuning per draw and checkpoint
  analysis/{org}.json, {org}_per_step.npz          40_analyse.py
  logs/                                              run logs
```
