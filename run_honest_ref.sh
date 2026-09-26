#!/usr/bin/env bash
# run_honest_ref.sh [--tiny] CFG:ORGANISM [CFG:ORGANISM ...]
# The honest graft (graft with HONEST activations): a comparison column of Table 2, not part of ActFlow. It needs the honest context (the honest
# instruction for prompt locks, the password for LoRA locks), so it runs only on locked models whose honest activations
# we have (captures/{organism}/honest_pool.npz, 16_honest_captures.py). ActFlow itself never reads its outputs.
# One cell after another on one GPU (pin it with CUDA_VISIBLE_DEVICES). Per cell:
#   1. copy the cell's captures into a staging root honestref_{cfg}/ (35_graft.py writes graft/ and landing/ under
#      ACTFLOW_RESULTS, and these must never land in results*/graft/, which 40_analyse.py reads as ActFlow rules),
#   2. write the stand-in landings (36_honest_reference.py),
#   3. graft all three (k = 4, 10, 40) with one call of the unchanged 35_graft.py (one model load),
#   4. copy the graft records to their home next to graft/:
#        results_{cfg}/honest-graft/{organism}/honestk_N0_k{k}/k{k}_d{d}.json
#        results_{cfg}/logs/honest_graft_{organism}.log
#      The staging root (captures copy, landings) stays on the pod and is not needed afterwards.
# Every cell's log ends with "== ... done ..." or "== FAILED ...". A failed cell stops this GPU's list (set -e).
# --tiny: CPU dry run on the tiny model. use_cfg forces results_tiny_{cfg}/ in tiny mode, so the dry run grafts there
# (clean honest* out of results_tiny_{cfg}/{landing,graft}/prompt/ afterwards) and step 4 copies from there.
# Example: CUDA_VISIBLE_DEVICES=0 bash run_honest_ref.sh qwen7b:prompt qwen7b:ft_lock
set -euo pipefail
cd "$(dirname "$0")"
TINY=""; if [ "${1:-}" = --tiny ]; then TINY=--tiny; shift; fi
if [ -n "$TINY" ]; then KS="2 4"; else KS="4 10 40"; fi        # draw sizes; add 1 here for k = 1 (80 draws per reference)
CFG=""; ORG=""; LOG=/dev/null
trap 'echo "== $(date -u +%FT%TZ) FAILED cfg=$CFG organism=$ORG" | tee -a "$LOG"' ERR
for cell in "$@"; do
  CFG=${cell%%:*}; ORG=${cell#*:}
  if [ -n "$TINY" ]; then SRC=results_tiny_$CFG; else SRC=results_$CFG; fi
  DST=honestref${TINY:+_tiny}_$CFG
  mkdir -p "$DST/captures/$ORG" "$DST/logs"; LOG="$DST/logs/$ORG.log"
  for f in pool.npz test.npz test_obqa.npz honest_pool.npz honest.json; do
    [ -f "$SRC/captures/$ORG/$f" ] || { echo "missing $SRC/captures/$ORG/$f"; false; }
    cp -p "$SRC/captures/$ORG/$f" "$DST/captures/$ORG/$f"
  done
  echo "== $(date -u +%FT%TZ) cfg=$CFG organism=$ORG root=$DST gpu=${CUDA_VISIBLE_DEVICES:-all}" | tee -a "$LOG"
  export ACTFLOW_RESULTS=$DST
  python scripts/36_honest_reference.py $TINY --cfg "$CFG" --organism "$ORG" --k $KS 2>&1 | tee -a "$LOG"
  LANDINGS=(); for k in $KS; do LANDINGS+=("honestk_N0_k$k"); done
  python scripts/35_graft.py $TINY --cfg "$CFG" --organism "$ORG" --batch 16 --landings "${LANDINGS[@]}" 2>&1 | tee -a "$LOG"
  unset ACTFLOW_RESULTS
  if [ -n "$TINY" ]; then GSRC=$SRC/graft/$ORG; else GSRC=$DST/graft/$ORG; fi
  mkdir -p "$SRC/honest-graft/$ORG" "$SRC/logs"
  for L in "${LANDINGS[@]}"; do cp -Rp "$GSRC/$L" "$SRC/honest-graft/$ORG/"; done
  cp -p "$GSRC/meta.json" "$SRC/honest-graft/$ORG/meta.json"
  echo "== $(date -u +%FT%TZ) done cfg=$CFG organism=$ORG -> $SRC/honest-graft/$ORG" | tee -a "$LOG"
  cp -p "$LOG" "$SRC/logs/honest_graft_$ORG.log"
done
