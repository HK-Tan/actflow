#!/usr/bin/env bash
# run_queue.sh --make SET:CFG:ORG ... > FILE          write the queue of these cells: their preps, then their slices
# run_queue.sh --queue FILE --gpus n | --devices 2,5   run the queue on this pod: one worker per GPU until FILE is empty
# Slice-level scheduling for many B / C cells on one pod. run_all.sh --gpus runs one
# cell at a time, so its GPUs idle while the cell's longest slice finishes; here every free GPU takes the next slice of any
# cell. A queue line is `WEIGHT CFG ORG run_A.sh-args` (run_all.sh --emit), or `0 CFG ORG --prep-only`. --make puts the
# prep line of every cell first (all gates of 00_check / 15_honest are known within minutes), then the cells in the order
# given, each longest slice first, so cells finish one after the other and a cell nobody has started can be moved to another
# pod. It scales the Llama / Mistral weights by 1.3 (32 layers of width 4096 against 28 of 3584); `sort -s -k1,1nr` of the
# slice lines would give longest-first across cells instead.
# A worker pops the first line under a lock (FILE.lock), runs the prep of that cell if it is not done yet (run_A.sh
# --prep-only, under a per-cell lock, so the cell's other workers wait for it and then find its files), then the slice
# (run_A.sh --no-prep). A failed prep writes $RES/logs/prep_failed_ORG and the workers drop that cell's remaining slices.
# A failed slice is reported and the worker goes on; the run exits 1 at the end if anything failed. A worker that finds
# FILE empty stops, so fill FILE before the start (append later cells while it runs, not after).
# --devices 2,5 runs workers on those CUDA devices only (GPUs another job has freed); --gpus n means --devices 0,..,n-1.
# Lines can be removed from or appended to FILE while it runs: take the same lock, e.g.
#   flock /workspace/queue.txt.lock sed -i '/ ft_lock_s5 /d' /workspace/queue.txt
# No analysis here: once every slice of a cell is done, run 40_analyse.py --cfg CFG --organism ORG.
# Examples:  bash run_queue.sh --make B:llama8b:ft_lock B:mistral7b:prompt C:qwen7b:ft_lock_s2 > /workspace/queue.txt
#            nohup bash run_queue.sh --queue /workspace/queue.txt --gpus 6 > /workspace/queue.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
if [ "${1:-}" = --make ]; then
  shift
  for c in "$@"; do
    IFS=: read -r s cfg org <<< "$c"
    bash run_all.sh --set "$s" --cfg "$cfg" --organism "$org" --emit > /dev/null || exit 2
    echo "0 $cfg $org --prep-only"
  done
  for c in "$@"; do
    IFS=: read -r s cfg org <<< "$c"
    bash run_all.sh --set "$s" --cfg "$cfg" --organism "$org" --emit
  done | awk '{ if ($2 != "qwen7b") $1 = int($1 * 1.3 + 0.5); print }'
  exit 0
fi
Q=""; GPUS=0; DEVS=""
while [ $# -gt 0 ]; do case "$1" in
  --queue) Q=$2; shift 2;; --gpus) GPUS=$2; shift 2;; --devices) DEVS=$2; shift 2;;
  *) echo "unknown arg $1"; exit 2;; esac; done
[ -n "$Q" ] && [ -f "$Q" ] || { echo "need --queue FILE (an existing file)"; exit 2; }
if [ -n "$DEVS" ]; then IFS=, read -r -a DEV <<< "$DEVS"; else DEV=(); for ((g = 0; g < GPUS; g++)); do DEV[g]=$g; done; fi
[ "${#DEV[@]}" -gt 0 ] || { echo "need --gpus n or --devices LIST"; exit 2; }
NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
for d in "${DEV[@]}"; do [[ "$d" =~ ^[0-9]+$ ]] && [ "$d" -lt "${NGPU:-0}" ] || { echo "no CUDA device $d (nvidia-smi -L lists ${NGPU:-0} GPUs); nothing started"; exit 2; }; done
ts() { date -u +%FT%TZ; }
res() { echo "${ACTFLOW_RESULTS:-results_$1}"; }
# the first non-blank line of FILE, removed from it; exit 1 when FILE is empty
pop() { ( flock 8; sed -i '/^[[:space:]]*$/d' "$Q"; l=$(head -n 1 "$Q"); [ -n "$l" ] || exit 1; sed -i 1d "$Q"; printf '%s\n' "$l" ) 8> "$Q.lock"; }
worker() {
  local g=$1 line w cfg org args r bad=0
  while line=$(pop); do
    read -r w cfg org args <<< "$line"
    r=$(res "$cfg"); mkdir -p "$r/logs"
    # prep once per cell; the lock makes the cell's other workers wait, then run_A.sh finds the prep files and skips them
    if ! ( flock 9
           [ ! -f "$r/logs/prep_failed_$org" ] || exit 1
           CUDA_VISIBLE_DEVICES=$g bash run_A.sh --cfg "$cfg" --organism "$org" --prep-only >> "$r/logs/queue_prep_$org.out" 2>&1 \
             || { touch "$r/logs/prep_failed_$org"; echo "== PREP FAILED $cfg $org on GPU $g $(ts): $r/logs/queue_prep_$org.out"; exit 1; }
         ) 9> "$r/logs/prep_$org.lock"; then
      echo "== GPU $g skips (prep of $cfg $org failed): $args"; bad=1; continue
    fi
    [ "$args" != --prep-only ] || { echo "== GPU $g prep ok $cfg $org $(ts)"; continue; }
    echo "== GPU $g start $cfg $org $args $(ts)"
    # shellcheck disable=SC2086
    if CUDA_VISIBLE_DEVICES=$g bash run_A.sh --cfg "$cfg" --organism "$org" --no-prep $args >> "$r/logs/queue_gpu$g.out" 2>&1; then
      echo "== GPU $g done  $cfg $org $args $(ts)"
    else echo "== GPU $g slice FAILED $cfg $org $args $(ts) (log $r/logs/queue_gpu$g.out)"; bad=1; fi
  done
  echo "== GPU $g: queue empty $(ts)"; return "$bad"
}
echo "== run_queue $Q on GPUs ${DEV[*]} $(ts): $(grep -c . "$Q") lines"
PIDS=()
for ((i = 0; i < ${#DEV[@]}; i++)); do
  [ "$i" -eq 0 ] || sleep "${QUEUE_STAGGER:-90}"       # stagger the first model loads in CPU RAM
  worker "${DEV[i]}" & PIDS[i]=$!
done
BAD=0; for ((i = 0; i < ${#DEV[@]}; i++)); do wait "${PIDS[i]}" || BAD=1; done
if [ "$BAD" = 1 ]; then echo "== run_queue FAILED somewhere (grep FAILED above) $(ts)"; exit 1; fi
echo "== run_queue all done $(ts)"
