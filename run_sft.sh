#!/usr/bin/env bash
# run_sft.sh --make > FILE            write the queue of 70_sft.py fine-tuning jobs (one line per cfg, organism, k; longest first)
# run_sft.sh --queue FILE --gpus n    one worker per GPU pops lines until FILE is empty (same pop-under-flock as run_queue.sh)
# A line is `WEIGHT CFG ORG ROW K`; ROW is sft. Weights = estimated L40S minutes, 4-min model load included.
# Cells: the 3 models x {prompt, ft_lock} (18 lines). SFT_CELLS="qwen7b:ft_lock_s2 ..." overrides.
# No prep: 70_sft.py needs only the model, the adapters and the ARC/OBQA items. Logs: results_{cfg}/logs/sft_{org}_{row}_k{k}.log
# A failed line is appended to FILE.failed; `cat FILE.failed >> FILE` requeues it and 70_sft.py resumes from its partial json.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}   # less fragmentation near the 45 GiB top
cd "$(dirname "$0")"
CELLS="${SFT_CELLS:-qwen7b:ft_lock qwen7b:prompt llama8b:ft_lock llama8b:prompt mistral7b:ft_lock mistral7b:prompt}"
if [ "${1:-}" = --make ]; then
  for c in $CELLS; do IFS=: read -r cfg org <<< "$c"
    for k in 4 10 40; do echo "$(( k == 4 ? 60 : (k == 10 ? 26 : 10) )) $cfg $org sft $k"; done; done | sort -s -k1,1nr
  exit 0
fi
Q=""; GPUS=0
while [ $# -gt 0 ]; do case "$1" in --queue) Q=$2; shift 2;; --gpus) GPUS=$2; shift 2;; *) echo "unknown arg $1"; exit 2;; esac; done
[ -f "$Q" ] && [ "$GPUS" -gt 0 ] || { echo "need --queue FILE --gpus n"; exit 2; }
ts() { date -u +%FT%TZ; }
pop() { ( flock 8; sed -i '/^[[:space:]]*$/d' "$Q"; l=$(head -n 1 "$Q"); [ -n "$l" ] || exit 1; sed -i 1d "$Q"; printf '%s\n' "$l" ) 8> "$Q.lock"; }
worker() {
  local g=$1 line w cfg org row k r bad=0
  while line=$(pop); do
    read -r w cfg org row k <<< "$line"
    r=results_$cfg; mkdir -p "$r/logs"
    echo "== GPU $g start $cfg $org $row k=$k $(ts)"
    if CUDA_VISIBLE_DEVICES=$g python scripts/70_sft.py --cfg "$cfg" --organism "$org" --rows "$row" --k "$k" >> "$r/logs/sft_${org}_${row}_k${k}.log" 2>&1
    then echo "== GPU $g done  $cfg $org $row k=$k $(ts)"; else echo "== GPU $g FAILED $cfg $org $row k=$k $(ts)"; ( flock 8; printf '%s\n' "$line" >> "$Q.failed" ) 8> "$Q.lock"; bad=1; fi
  done
  echo "== GPU $g: queue empty $(ts)"; return "$bad"
}
PIDS=(); for ((g = 0; g < GPUS; g++)); do [ "$g" -eq 0 ] || sleep "${QUEUE_STAGGER:-30}"; worker "$g" & PIDS[g]=$!; done
BAD=0; for p in "${PIDS[@]}"; do wait "$p" || BAD=1; done
[ "$BAD" = 0 ] && echo "== run_sft all done $(ts)" || { echo "== run_sft FAILED somewhere $(ts)"; exit 1; }
