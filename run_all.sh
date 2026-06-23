#!/usr/bin/env bash
set -euo pipefail
export NCCL_P2P_LEVEL=NVL
LOG=run_$(date +%Y%m%d_%H%M%S).log

# 1. Profile SFT (short, ~16 steps, throwaway output)
echo "=== PROFILE SFT ===" | tee -a "$LOG"
./profile_fsdp.sh sft 2>&1 | tee -a "$LOG"

# 2. Full SFT training
echo "=== TRAIN SFT ===" | tee -a "$LOG"
torchrun --nproc_per_node=4 train_fsdp_sft.py \
  --model JetBrains/Mellum2-12B-A2.5B-Thinking \
  --data data/sft_clean.jsonl --out ckpts/sft \
  --epochs 2 --max-seq-len 30100 --micro-bsz 1 --grad-accum 8 --lr 1e-5 \
  2>&1 | tee -a "$LOG"

# 3. Profile DPO (needs ckpts/sft from step 2)
echo "=== PROFILE DPO ===" | tee -a "$LOG"
./profile_fsdp.sh dpo 2>&1 | tee -a "$LOG"

# 4. Full DPO training
echo "=== TRAIN DPO ===" | tee -a "$LOG"
torchrun --nproc_per_node=4 train_fsdp_dpo.py \
  --policy ckpts/sft --data data/dolci_dpo_pairs.jsonl --out ckpts/dpo \
  --epochs 1 --max-seq-len 8192 --micro-bsz 1 --grad-accum 8 --lr 5e-7 --beta 0.1 \
  2>&1 | tee -a "$LOG"

echo "=== DONE ===" | tee -a "$LOG"