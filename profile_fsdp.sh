#!/usr/bin/env bash
# profile_fsdp.sh — scoped nsys capture for the FSDP training runs.
#
# Usage:
#   ./profile_fsdp.sh sft
#   ./profile_fsdp.sh dpo
#
# Key idea: we do NOT trace the whole multi-hour run (that produces a 100s-of-GB
# trace that won't open). Instead the training script calls cudaProfilerStart()
# at PROFILE_START_STEP and cudaProfilerStop() at PROFILE_STOP_STEP, and nsys is
# told to only capture between those calls via --capture-range=cudaProfilerApi.
# Result: a few-GB trace of ~5-10 representative steps.
#
# Pass profiling controls to the script via env vars (read in the Python).

set -euo pipefail
TASK="${1:?Usage: ./profile_fsdp.sh {sft|dpo}}"

# --- which steps to capture (steady state, AFTER warmup/allocator settles) ---
export PROFILE_ENABLE=1
export PROFILE_START_STEP="${PROFILE_START_STEP:-10}"
export PROFILE_STOP_STEP="${PROFILE_STOP_STEP:-16}"
export NSYS_OUT="nsys_${TASK}_$(date +%Y%m%d_%H%M%S)"

# --- nsys flags ---
# --trace: cuda (kernels/memops), nvtx (our named ranges), osrt (CPU/syscalls),
#          cudnn/cublas (GEMM libs). Add 'cuda-graph-trace=node' if graphs used.
# --gpu-metrics-devices=all: SM occupancy / DRAM / PCIe utilization sampling ->
#          this is what shows you the CPU-offload PCIe stalls.
# --capture-range: only record between cudaProfilerStart/Stop.
# --cuda-memory-usage=true: track allocations (helpful for the OOM story).
NSYS_ARGS=(
  profile
  --output="${NSYS_OUT}"
  --force-overwrite=true
  --trace=cuda,nvtx,osrt,cudnn,cublas
  --gpu-metrics-devices=all
  --capture-range=cudaProfilerApi
  --capture-range-end=stop
  --cuda-memory-usage=true
  --sample=cpu
)

# IMPORTANT: profile a SINGLE rank, not all 4. Wrapping torchrun in nsys profiles
# rank0's process tree; multi-rank traces overlap and bloat. One rank is enough
# to see kernel timeline + MoE routing + PCIe stalls. We pin to 1 visible GPU? No —
# FSDP needs all 4 ranks running. So: run torchrun normally but only nsys-wrap the
# launcher; nsys follows child processes. To keep size sane we rely on the narrow
# capture window above. (--nproc_per_node stays 4 so FSDP collectives work.)

if [[ "$TASK" == "sft" ]]; then
  nsys "${NSYS_ARGS[@]}" \
    torchrun --nproc_per_node=4 train_fsdp_sft.py \
      --model JetBrains/Mellum2-12B-A2.5B-Thinking \
      --data data/sft_clean.jsonl --out ckpts/sft_profile \
      --epochs 1 --max-seq-len 30100 --micro-bsz 1 --grad-accum 8 --lr 1e-5 \
      --log-every 1 --max-steps "${PROFILE_STOP_STEP}"
elif [[ "$TASK" == "dpo" ]]; then
  nsys "${NSYS_ARGS[@]}" \
    torchrun --nproc_per_node=4 train_fsdp_dpo.py \
      --policy ckpts/sft --data data/dolci_dpo_pairs.jsonl --out ckpts/dpo_profile \
      --epochs 1 --max-seq-len 8192 --micro-bsz 1 --grad-accum 8 \
      --lr 5e-7 --beta 0.1 \
      --log-every 1 --max-steps "${PROFILE_STOP_STEP}"
else
  echo "unknown task: $TASK"; exit 1
fi

echo "=== wrote ${NSYS_OUT}.nsys-rep ==="
echo "Inspect headless with:"
echo "  nsys stats ${NSYS_OUT}.nsys-rep"
echo "  nsys stats --report gpukernsum ${NSYS_OUT}.nsys-rep   # kernel time summary"
echo "  nsys stats --report gpumemsizesum ${NSYS_OUT}.nsys-rep # mem ops"
echo "Or open ${NSYS_OUT}.nsys-rep in the Nsight Systems GUI locally."
