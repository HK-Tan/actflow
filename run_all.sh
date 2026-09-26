#!/usr/bin/env bash
# run_all.sh [--cfg qwen7b|llama8b|mistral7b] --organism ORG [--set A|B|C] [--shard i/n | --gpus n] [--list | --emit] [--no-prep] [--no-analysis]
# Every job slice of one organism for one block set on this machine, or this machine's share of them.
#   --set A (default): the 16 block-A slices for prompt / ft_lock.  --set C: the 6 slices of one C cell (ft_lock_s2..s5);
#   --set B: the same 6 slices with --block B, with --cfg llama8b / mistral7b, for the ft_lock and prompt cells
#   of that model. A and C are Qwen only (configs/blocks.yaml). --cfg (default qwen7b) reaches every run_A.sh and python
#   call; results go to $ACTFLOW_RESULTS, else results_{cfg}/ (io.use_cfg).
#   --gpus n: one pod with n GPUs (checked against nvidia-smi -L first). Prep runs once on GPU 0, then the n shards start
#   together, one per GPU (CUDA_VISIBLE_DEVICES), 90 s apart so the model loads do not pile up in CPU RAM; the analysis
#   runs when all are done.
#   --emit: print every slice as one queue line `WEIGHT CFG ORG run_A.sh-args` and exit (run_queue.sh --make uses it).
#   The 16 slices below are the runs of blocks A1-A3 with the k = 1 AF job split by rule. Weights are the
#   estimated A40 hours x 10. With --shard i/n the slices are dealt to n pods longest-first, each to the least-loaded pod,
#   and this pod runs share i (0-based); pods 0/n .. (n-1)/n together cover every slice exactly once.
#   Each slice is one `run_A.sh --no-prep` call. Prep (check, captures, honest) runs once per pod before the slices; a failed
#   prep stops the pod. A failed slice is reported and the pod's other slices still run; the pod then exits 1 and names it.
#   Without --shard (one pod does everything) 40_analyse.py runs at the end; with --shard it does not,
#   because the analysis needs every pod's results root merged on one machine (rsync them back, then run them on the laptop).
# Examples:  bash run_all.sh --organism prompt --list                   # print the plan, run nothing
#            bash run_all.sh --organism prompt --shard 0/8 --list        # what pod 0 of 8 would run
#            nohup bash run_all.sh --organism ft_lock --shard 3/8 > /workspace/shard3.out 2>&1 &
#            nohup bash run_all.sh --organism prompt --gpus 8 > /workspace/all.out 2>&1 &     # one 8 x A40 pod
#            nohup bash run_all.sh --set C --organism ft_lock_s2 --gpus 6 > /workspace/c2.out 2>&1 &   # one C cell, 6 GPUs
#            nohup bash run_all.sh --set B --cfg llama8b --organism ft_lock --gpus 6 > /workspace/b_llama_ft.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
CFG=qwen7b; ORG=""; SET=A; SHARD=""; GPUS=0; LIST=0; EMIT=0; PREP=1; ANALYSIS=1
while [ $# -gt 0 ]; do case "$1" in
  --cfg) CFG=$2; shift 2;; --organism) ORG=$2; shift 2;; --set) SET=$2; shift 2;; --shard) SHARD=$2; shift 2;; --gpus) GPUS=$2; shift 2;; --list) LIST=1; shift;; --emit) EMIT=1; shift;;
  --no-prep) PREP=0; shift;; --no-analysis) ANALYSIS=0; shift;;
  *) echo "unknown arg $1"; exit 2;; esac; done
[ -n "$ORG" ] || { echo "need --organism"; exit 2; }
[ "$GPUS" -eq 0 ] || [ -z "$SHARD" ] || { echo "--gpus and --shard exclude each other"; exit 2; }
# the cells of each set and cfg, as configs/blocks.yaml resolves them (_common.blocks_cfg)
case "$SET/$CFG" in
  A/qwen7b) CELLS="prompt ft_lock";; C/qwen7b) CELLS="ft_lock_s2 ft_lock_s3 ft_lock_s4 ft_lock_s5";;
  B/llama8b|B/mistral7b) CELLS="ft_lock prompt";;
  B/qwen7b) echo "--set B is Llama and Mistral only (configs/blocks.yaml); Qwen has --set A and C"; exit 2;;
  A/llama8b|A/mistral7b|C/llama8b|C/mistral7b) echo "--set $SET is Qwen only (configs/blocks.yaml); --cfg $CFG has --set B"; exit 2;;
  *) echo "unknown --set $SET (A, B or C) or --cfg $CFG (qwen7b, llama8b or mistral7b)"; exit 2;;
esac
[[ " $CELLS " == *" $ORG "* ]] || { echo "--set $SET --cfg $CFG has no cell $ORG (cells: $CELLS)"; exit 2; }
RES=${ACTFLOW_RESULTS:-results_$CFG}
# weight (estimated A40 h x 10; GD grafts only its final state) | run_A.sh arguments.  Longest first.
JOBS_A=(
  "54|--block A1 --rules af af01 afpr af5 --k 40"
  "48|--block A1 --rules af af01 afpr af5 --k 4"
  "41|--block A1 --rules af af01 afpr af5 --k 10"
  "39|--block A2 --k 40"
  "35|--block A2 --k 4"
  "31|--block A1 --rules gd --k 1"
  "30|--block A1 --rules af --k 1"
  "30|--block A1 --rules af01 --k 1"
  "30|--block A1 --rules afpr --k 1"
  "30|--block A2 --k 10"
  "22|--block A3 --k 40"
  "20|--block A1 --rules gd --k 40"
  "20|--block A1 --rules gd --k 4"
  "20|--block A3 --k 4"
  "19|--block A1 --rules gd --k 10"
  "18|--block A3 --k 10"
)
# one B or C cell: GD per k, then the three AF rules per k (weights = estimated A40 h x 10;
# Qwen estimates, also used for the Llama / Mistral cells: 32 layers of width 4096 against 28 of 3584, so those run longer)
JOBS_BC=(
  "20|--block SET --rules gd --k 40"
  "20|--block SET --rules gd --k 4"
  "19|--block SET --rules gd --k 10"
  "17|--block SET --rules af afpr af5 --k 40"
  "15|--block SET --rules af afpr af5 --k 4"
  "13|--block SET --rules af afpr af5 --k 10"
)
case "$SET" in
  A) JOBS=("${JOBS_A[@]}");;
  B|C) JOBS=("${JOBS_BC[@]/SET/$SET}");;
esac
if [ "$EMIT" = 1 ]; then for j in "${JOBS[@]}"; do echo "${j%%|*} $CFG $ORG ${j#*|}"; done; exit 0; fi
I=0; N=1
if [ -n "$SHARD" ]; then I=${SHARD%/*}; N=${SHARD#*/}; [ "$I" -ge 0 ] && [ "$I" -lt "$N" ] || { echo "bad --shard $SHARD"; exit 2; }; fi
[ "$GPUS" -eq 0 ] || N=$GPUS
LOAD=(); for ((p = 0; p < N; p++)); do LOAD[p]=0; done
MINE=(); PLAN=()
for j in "${JOBS[@]}"; do
  w=${j%%|*}; a=${j#*|}; best=0
  for ((p = 1; p < N; p++)); do [ "${LOAD[p]}" -lt "${LOAD[best]}" ] && best=$p; done
  LOAD[best]=$(( LOAD[best] + w )); PLAN+=("$best|$w|$a"); [ "$best" -eq "$I" ] && MINE+=("$a")
done
if [ "$GPUS" -gt 0 ]; then echo "== run_all cfg=$CFG set=$SET organism=$ORG: all ${#JOBS[@]} slices on $GPUS GPUs of this pod"
else echo "== run_all cfg=$CFG set=$SET organism=$ORG shard=$I/$N: ${#MINE[@]} of ${#JOBS[@]} slices, about $(( LOAD[I] / 10 )).$(( LOAD[I] % 10 )) A40 h + prep"; fi
for ((p = 0; p < N; p++)); do
  echo "  pod $p/$N  ~$(( LOAD[p] / 10 )).$(( LOAD[p] % 10 )) h:"; for e in "${PLAN[@]}"; do [ "${e%%|*}" = "$p" ] && echo "      run_A.sh --cfg $CFG --organism $ORG ${e#*|*|}"; done
done
[ "$LIST" = 0 ] || exit 0
mkdir -p "$RES/logs"                                 # after --list, so listing writes nothing
if [ "$GPUS" -gt 0 ]; then
  # a shard whose GPU index does not exist sees no CUDA; count the GPUs before anything starts
  NGPU=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
  [ "$NGPU" -ge "$GPUS" ] || { echo "--gpus $GPUS but nvidia-smi -L lists ${NGPU:-0} GPUs on this pod; nothing started"; exit 2; }
  [ "$PREP" = 0 ] || CUDA_VISIBLE_DEVICES=0 bash run_A.sh --cfg "$CFG" --organism "$ORG" --prep-only
  PIDS=()
  for ((g = 0; g < GPUS; g++)); do
    [ "$g" -eq 0 ] || sleep 90
    CUDA_VISIBLE_DEVICES=$g bash run_all.sh --cfg "$CFG" --set "$SET" --organism "$ORG" --shard "$g/$GPUS" --no-prep --no-analysis > "$RES/logs/${ORG}_shard${g}of${GPUS}.out" 2>&1 &
    PIDS[g]=$!; echo "  started shard $g/$GPUS on GPU $g (pid $!) -> $RES/logs/${ORG}_shard${g}of${GPUS}.out"
  done
  # wait for each shard by pid: a bare `wait` always returns 0 and would hide a failed shard
  BAD=()
  for ((g = 0; g < GPUS; g++)); do wait "${PIDS[g]}" || BAD+=("$g"); done
  if [ "${#BAD[@]}" -gt 0 ]; then
    echo "== FAILED: ${#BAD[@]} of $GPUS shards. No analysis. $(date -u +%FT%TZ)"
    for g in "${BAD[@]}"; do echo "     shard $g: $RES/logs/${ORG}_shard${g}of${GPUS}.out"; done
    exit 1
  fi
  echo "== all $GPUS shards done $(date -u +%FT%TZ)"
else
  # prep once; a failed prep (00_check, captures or honest) stops the shard here
  [ "$PREP" = 0 ] || bash run_A.sh --cfg "$CFG" --organism "$ORG" --prep-only
  # a failed slice does not stop the pod's other slices (they are independent); the shard exits 1 at the end and names them
  FAILED=()
  for a in "${MINE[@]}"; do
    # shellcheck disable=SC2086
    bash run_A.sh --cfg "$CFG" --organism "$ORG" --no-prep $a || { echo "== slice FAILED: run_A.sh --cfg $CFG --organism $ORG $a"; FAILED+=("$a"); }
  done
  if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "== shard $I/$N FAILED: ${#FAILED[@]} of ${#MINE[@]} slices failed (logs in $RES/logs/). No analysis. $(date -u +%FT%TZ)"
    for a in "${FAILED[@]}"; do echo "     run_A.sh --cfg $CFG --organism $ORG $a"; done
    exit 1
  fi
fi
if { [ "$N" -eq 1 ] || [ "$GPUS" -gt 0 ]; } && [ "$ANALYSIS" = 1 ]; then
  python scripts/40_analyse.py --cfg "$CFG" --organism "$ORG"
else
  echo "== shard $I/$N done $(date -u +%FT%TZ). rsync $RES/ back and run 40_analyse.py --cfg $CFG once every shard is in."
fi
