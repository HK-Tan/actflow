#!/usr/bin/env bash
# run_A.sh [--cfg qwen7b|llama8b|mistral7b] --organism ORG --block A1|A2|A3|B|C [--rules ...] [--k ...] [--N ...] [--tag NAME]
#          [--item-rows final|all|none] [--no-prep | --prep-only]
# One job = the flows of one slice of a block, then the grafts of exactly the landings that slice produced, on one GPU.
# Slices are independent, so any number of them run at once on different GPUs.
#   prep (once per organism per machine, ~5 min): 00_check -> 10_captures -> 15_honest
#   flows: 30_manufacture.py --block ... -> landing/{organism}/*.npz + produced_{tag}.txt
#   grafts: 35_graft.py --landings $(cat produced_{tag}.txt)
# --cfg (default qwen7b) picks configs/{cfg}.yaml and reaches every python call; results go to $ACTFLOW_RESULTS, else
# results_{cfg}/ (the rule of io.use_cfg).
# Examples:  bash run_A.sh --organism prompt --block A1 --rules af af01 afpr af5 --k 1
#            bash run_A.sh --organism prompt --block A1 --rules gd --k 40
#            bash run_A.sh --organism ft_lock --block A2 --k 10
#            bash run_A.sh --cfg llama8b --organism prompt --block B --rules gd --k 40
set -euo pipefail
cd "$(dirname "$0")"
CFG=qwen7b; ORG=""; BLOCK=""; TAG=""; ITEM_ROWS=final; PREP=1; PREP_ONLY=0; EXTRA=()
while [ $# -gt 0 ]; do case "$1" in
  --cfg) CFG=$2; shift 2;; --organism) ORG=$2; shift 2;; --block) BLOCK=$2; shift 2;; --tag) TAG=$2; shift 2;; --item-rows) ITEM_ROWS=$2; shift 2;;
  --no-prep) PREP=0; shift;; --prep-only) PREP_ONLY=1; BLOCK=${BLOCK:-prep}; shift;;
  --rules|--k|--N) F=$1; shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do EXTRA+=("$F" "$1"); F=""; shift; done;;
  *) echo "unknown arg $1"; exit 2;; esac; done
[ -n "$ORG" ] && [ -n "$BLOCK" ] || { echo "need --organism and --block"; exit 2; }
# --rules a b c must reach python as one flag with several values
ARGS=(); prev=""; for x in ${EXTRA[@]+"${EXTRA[@]}"}; do if [ -n "$x" ]; then ARGS+=("$x"); fi; done
# the tag keeps the separators, so --N 1 8 (N_1_8) and --N 18 (N_18) cannot share a tag
[ -n "$TAG" ] || TAG="${BLOCK}_$(echo "${ARGS[*]:-}" | sed 's/--//g; s/ /_/g')"; [ "$TAG" != "${BLOCK}_" ] || TAG="$BLOCK"
RES=${ACTFLOW_RESULTS:-results_$CFG}
mkdir -p "$RES/logs"; LOG="$RES/logs/${ORG}_${TAG}.log"
echo "== run_A cfg=$CFG organism=$ORG block=$BLOCK args=${ARGS[*]:-} tag=$TAG $(date -u +%FT%TZ)" | tee -a "$LOG"
run() { echo "+ $*" | tee -a "$LOG"; python "$@" 2>&1 | tee -a "$LOG"; }
CAP="$RES/captures/$ORG"
if [ "$PREP" = 1 ]; then
  [ -f "$RES/logs/check_$ORG.json" ] || run scripts/00_check.py --cfg "$CFG" --organism "$ORG" --item 0 --layers 10 20 --mbatch 40
  { [ -f "$CAP/pool.npz" ] && [ -f "$CAP/test.npz" ] && [ -f "$CAP/test_obqa.npz" ]; } || run scripts/10_captures.py --cfg "$CFG" --organism "$ORG" --batch 16
  [ -f "$CAP/honest.json" ] || run scripts/15_honest.py --cfg "$CFG" --organism "$ORG" --batch 16
fi
[ "$PREP_ONLY" = 0 ] || { echo "== prep done $ORG $(date -u +%FT%TZ)" | tee -a "$LOG"; exit 0; }
run scripts/30_manufacture.py --cfg "$CFG" --organism "$ORG" --block "$BLOCK" --batch 40 --tag "$TAG" ${ARGS[@]+"${ARGS[@]}"}
LANDINGS=(); while IFS= read -r ln; do [ -n "$ln" ] && LANDINGS+=("$ln"); done < "$RES/landing/$ORG/produced_${TAG}.txt"
if [ "${#LANDINGS[@]}" -eq 0 ]; then echo "no landings produced by this slice; nothing to graft" | tee -a "$LOG"
else run scripts/35_graft.py --cfg "$CFG" --organism "$ORG" --batch 16 --item-rows "$ITEM_ROWS" --landings "${LANDINGS[@]}"; fi
echo "== done $TAG $(date -u +%FT%TZ)" | tee -a "$LOG"
